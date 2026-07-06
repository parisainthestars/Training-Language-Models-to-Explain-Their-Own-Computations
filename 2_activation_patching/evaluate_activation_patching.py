"""
Evaluation script for activation patching explainer models.

This script evaluates explainer models on activation patching tasks and computes:
- Exact Match: Full explanation matches exactly
- Has-Changed F1: Macro F1 score for predicting whether output changed
- Content Match: Whether predicted token matches ground truth token

Supports ablation studies by removing activation, layer, or token information.

Usage:
    # Evaluate trained explainer model
    python evaluate_activation_patching.py \
        --explainer_model ./explainer_checkpoints \
        --test_data activation_patch_dataset.json \
        --run_ablations \
        --num_runs 5
    
    # Evaluate untrained baseline
    python evaluate_activation_patching.py \
        --explainer_model Qwen/Qwen3-8B \
        --test_data activation_patch_dataset.json \
        --baseline_only \
        --num_runs 5
"""

import json
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
import argparse
try:
    from sklearn.metrics import f1_score
except ImportError:
    f1_score = None

# Import from activation_patching.py
from activation_patching import (
    compute_has_changed_f1,
    compute_content_match,
    compute_exact_match,
    find_v_index,
)


@dataclass
class PredictionResult:
    """Structured prediction result."""
    has_changed: bool
    next_token: str
    explanation: str
    raw_output: str


def parse_explanation(explanation_text: str) -> Optional[PredictionResult]:
    """
    Parse explanation text to extract has_changed, next_token, and full explanation.
    
    Expected formats:
    - "The output would change to <<<TOKEN>>>."
    - "The output would remain unchanged from <<<TOKEN>>>."
    """
    explanation_text = explanation_text.strip()
    
    # Pattern for "change to"
    change_pattern = r"the output would change to\s*<<<([^>]+)>>>\.?"
    match = re.search(change_pattern, explanation_text, re.IGNORECASE)
    if match:
        token = match.group(1).strip()
        return PredictionResult(
            has_changed=True,
            next_token=token,
            explanation=explanation_text,
            raw_output=explanation_text
        )
    
    # Pattern for "remain unchanged"
    unchanged_pattern = r"the output would remain unchanged from\s*<<<([^>]+)>>>\.?"
    match = re.search(unchanged_pattern, explanation_text, re.IGNORECASE)
    if match:
        token = match.group(1).strip()
        return PredictionResult(
            has_changed=False,
            next_token=token,
            explanation=explanation_text,
            raw_output=explanation_text
        )
    
    # Fallback: try to extract any token in <<<...>>>
    token_pattern = r"<<<([^>]+)>>>"
    match = re.search(token_pattern, explanation_text, re.IGNORECASE)
    if match:
        token = match.group(1).strip()
        # Try to infer has_changed from keywords
        has_changed = "change" in explanation_text.lower() and "unchanged" not in explanation_text.lower()
        return PredictionResult(
            has_changed=has_changed,
            next_token=token,
            explanation=explanation_text,
            raw_output=explanation_text
        )
    
    # If no pattern matches, return None
    return None


def generate_explanation(
    model,
    tokenizer,
    question: str,
    feature_vector: List[float],
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    do_sample: bool = False
) -> str:
    """
    Generate explanation using the explainer model.
    
    Args:
        model: Explainer model
        tokenizer: Tokenizer
        question: Input question with [s]v[e] tokens
        feature_vector: Feature vector v to inject at [s]v[e] position
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
    
    Returns:
        Generated explanation text
    """
    # Tokenize question
    enc = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    
    # Find v index
    try:
        v_idx = find_v_index(input_ids[0].tolist(), tokenizer)
    except Exception as e:
        # Fallback: find [s] token and use next position
        s_id = tokenizer.convert_tokens_to_ids("[s]")
        s_positions = (input_ids[0] == s_id).nonzero(as_tuple=True)[0]
        if len(s_positions) > 0:
            v_idx = int(s_positions[0].item()) + 1
        else:
            v_idx = len(input_ids[0]) // 2
    
    # Prepare feature vector
    v = torch.tensor(feature_vector, dtype=torch.float32)
    v = v / (v.norm() + 1e-8)  # Normalize
    
    # Get embeddings and inject v
    emb_layer = model.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids)
    
    # Inject v at v_idx
    if v_idx < inputs_embeds.size(1):
        inputs_embeds[0, v_idx, :] = v.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    prompt_len = input_ids.shape[1]
    generated_ids = outputs[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return generated_text


def create_ablated_question(
    question: str,
    remove_activation: bool = False,
    remove_layer: bool = False,
    remove_token: bool = False
) -> str:
    """
    Create ablated version of question by removing specified information.
    
    Args:
        question: Original question
        remove_activation: Remove [s]v[e] feature vector information
        remove_layer: Remove layer information
        remove_token: Remove token position information
    
    Returns:
        Ablated question
    """
    ablated = question
    
    # Remove activation information ([s]v[e])
    if remove_activation:
        ablated = re.sub(r'\[s\]v\[e\]', 'feature', ablated)
        # Also remove references to feature vector
        ablated = re.sub(r'feature\s+\[s\]v\[e\]', 'feature', ablated, flags=re.IGNORECASE)
    
    # Remove layer information
    if remove_layer:
        ablated = re.sub(r'at layer \d+', 'at layer ℓ', ablated, flags=re.IGNORECASE)
        ablated = re.sub(r'layer \d+', 'layer ℓ', ablated, flags=re.IGNORECASE)
    
    # Remove token position information
    if remove_token:
        ablated = re.sub(r'into token \d+', 'into token t', ablated, flags=re.IGNORECASE)
        ablated = re.sub(r'token \d+', 'token t', ablated, flags=re.IGNORECASE)
        ablated = re.sub(r'at tokens \d+', 'at tokens t', ablated, flags=re.IGNORECASE)
    
    return ablated


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    ablation_config: Optional[Dict[str, bool]] = None,
    batch_size: int = 1,
    max_new_tokens: int = 50,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate explainer model on test data.
    
    Args:
        model: Explainer model
        tokenizer: Tokenizer
        test_data: List of test samples with keys: question, answer, has_changed, 
                   original_next_token, patched_next_token, feature_vector
        ablation_config: Dict with keys: remove_activation, remove_layer, remove_token
        batch_size: Batch size for generation (currently supports 1)
        max_new_tokens: Maximum tokens to generate
        verbose: Whether to print progress
    
    Returns:
        Dictionary with metrics and predictions
    """
    if ablation_config is None:
        ablation_config = {
            "remove_activation": False,
            "remove_layer": False,
            "remove_token": False
        }
    
    predictions = []
    ground_truth = []
    
    iterator = tqdm(test_data, desc="Evaluating") if verbose else test_data
    
    for item in iterator:
        # Get question
        question = item["question"]
        
        # Apply ablation if specified
        if any(ablation_config.values()):
            question = create_ablated_question(
                question,
                remove_activation=ablation_config.get("remove_activation", False),
                remove_layer=ablation_config.get("remove_layer", False),
                remove_token=ablation_config.get("remove_token", False)
            )
        
        # Get feature vector
        feature_vector = item["feature_vector"]
        
        # Generate explanation
        try:
            generated_text = generate_explanation(
                model, tokenizer, question, feature_vector,
                max_new_tokens=max_new_tokens
            )
            
            # Parse prediction
            pred_result = parse_explanation(generated_text)
            
            if pred_result is None:
                # Fallback: create minimal prediction
                pred_result = PredictionResult(
                    has_changed=False,
                    next_token="",
                    explanation=generated_text,
                    raw_output=generated_text
                )
        except Exception as e:
            if verbose:
                print(f"Error generating explanation: {e}")
            pred_result = PredictionResult(
                has_changed=False,
                next_token="",
                explanation="",
                raw_output=""
            )
        
        # Get ground truth
        gt_has_changed = item["has_changed"]
        gt_token = item["patched_next_token"] if gt_has_changed else item["original_next_token"]
        gt_explanation = item["answer"]
        
        # Store predictions and ground truth
        predictions.append({
            "has_changed": pred_result.has_changed,
            "next_token": pred_result.next_token,
            "explanation": pred_result.explanation,
            "raw_output": pred_result.raw_output
        })
        
        ground_truth.append({
            "has_changed": gt_has_changed,
            "next_token": gt_token,
            "explanation": gt_explanation
        })
    
    # Compute metrics
    pred_has_changed = [p["has_changed"] for p in predictions]
    gt_has_changed = [g["has_changed"] for g in ground_truth]
    
    pred_tokens = [p["next_token"] for p in predictions]
    gt_tokens = [g["next_token"] for g in ground_truth]
    
    pred_explanations = [p["explanation"] for p in predictions]
    gt_explanations = [g["explanation"] for g in ground_truth]
    
    # Compute metrics
    exact_match = compute_exact_match(pred_explanations, gt_explanations)
    has_changed_f1 = compute_has_changed_f1(pred_has_changed, gt_has_changed)
    content_match = compute_content_match(pred_tokens, gt_tokens)
    
    return {
        "exact_match": exact_match,
        "has_changed_f1": has_changed_f1,
        "content_match": content_match,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "num_samples": len(test_data)
    }


def run_evaluation_with_bootstrap(
    model,
    tokenizer,
    test_data: List[Dict],
    ablation_config: Optional[Dict[str, bool]] = None,
    num_runs: int = 5,
    bootstrap_samples: int = 1000,
    verbose: bool = False,
    use_baseline: bool = False,
    max_new_tokens: int = 50
) -> Dict[str, Tuple[float, float]]:
    """
    Run evaluation multiple times with bootstrap sampling for statistical significance.
    
    Args:
        model: Explainer model
        tokenizer: Tokenizer
        test_data: Test data
        ablation_config: Ablation configuration
        num_runs: Number of evaluation runs
        bootstrap_samples: Number of bootstrap samples
        verbose: Whether to print progress
    
    Returns:
        Dictionary with metric names as keys and (mean, std_error) tuples as values
    """
    all_results = []
    
    for run_idx in range(num_runs):
        if verbose:
            print(f"Run {run_idx + 1}/{num_runs}")
        
        # Bootstrap sample
        if bootstrap_samples > 0 and len(test_data) > 0:
            indices = np.random.choice(len(test_data), size=len(test_data), replace=True)
            sampled_data = [test_data[i] for i in indices]
        else:
            sampled_data = test_data
        
        # Evaluate
        if use_baseline:
            results = evaluate_baseline_model(
                model, tokenizer, sampled_data,
                max_new_tokens=max_new_tokens,
                verbose=verbose and run_idx == 0
            )
        else:
            results = evaluate_model(
                model, tokenizer, sampled_data,
                ablation_config=ablation_config,
                max_new_tokens=max_new_tokens,
                verbose=verbose and run_idx == 0  # Only verbose on first run
            )
        
        all_results.append({
            "exact_match": results["exact_match"],
            "has_changed_f1": results["has_changed_f1"],
            "content_match": results["content_match"]
        })
    
    # Compute mean and standard error
    metrics = ["exact_match", "has_changed_f1", "content_match"]
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        mean = np.mean(values)
        std_error = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[metric] = (mean, std_error)
    
    return summary


def format_results_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    model_name: str = "Model"
) -> str:
    """
    Format evaluation results as a table.
    
    Args:
        results: Dictionary with configuration names as keys and metric dicts as values
        model_name: Name of the model being evaluated
    
    Returns:
        Formatted table string
    """
    # Convert to percentage
    def format_metric(mean: float, std_error: float) -> str:
        mean_pct = mean * 100.0
        std_error_pct = std_error * 100.0
        return f"{mean_pct:.1f}±{std_error_pct:.1f}"
    
    # Header
    header = f"{'Target':<20} {'Exact Match':<15} {'Has-Changed F1':<15} {'Content Match':<15}"
    lines = [header, "=" * 65]
    
    # Add main model result
    if "full" in results:
        config_results = results["full"]
        line = f"{model_name:<20}"
        line += f"{format_metric(*config_results['exact_match']):<15}"
        line += f"{format_metric(*config_results['has_changed_f1']):<15}"
        line += f"{format_metric(*config_results['content_match']):<15}"
        lines.append(line)
    elif "baseline" in results:
        config_results = results["baseline"]
        line = f"{model_name + ' (Untrained)':<20}"
        line += f"{format_metric(*config_results['exact_match']):<15}"
        line += f"{format_metric(*config_results['has_changed_f1']):<15}"
        line += f"{format_metric(*config_results['content_match']):<15}"
        lines.append(line)
    elif len(results) == 1:
        config_name = list(results.keys())[0]
        config_results = results[config_name]
        line = f"{model_name:<20}"
        line += f"{format_metric(*config_results['exact_match']):<15}"
        line += f"{format_metric(*config_results['has_changed_f1']):<15}"
        line += f"{format_metric(*config_results['content_match']):<15}"
        lines.append(line)
    
    # Add ablation results
    ablation_order = ["activation", "layer", "token"]
    for ablation in ablation_order:
        if ablation in results:
            config_results = results[ablation]
            line = f"{'– ' + ablation:<20}"
            line += f"{format_metric(*config_results['exact_match']):<15}"
            line += f"{format_metric(*config_results['has_changed_f1']):<15}"
            line += f"{format_metric(*config_results['content_match']):<15}"
            lines.append(line)
    
    return "\n".join(lines)


def evaluate_baseline_model(
    model,
    tokenizer,
    test_data: List[Dict],
    max_new_tokens: int = 50,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate untrained baseline model by prompting it directly.
    
    Args:
        model: Untrained baseline model
        tokenizer: Tokenizer
        test_data: Test data
        max_new_tokens: Maximum tokens to generate
        verbose: Whether to print progress
    
    Returns:
        Dictionary with metrics and predictions
    """
    predictions = []
    ground_truth = []
    
    iterator = tqdm(test_data, desc="Evaluating baseline") if verbose else test_data
    
    for item in iterator:
        # Get question (without injecting feature vector - just prompt)
        question = item["question"]
        
        # For baseline, we just prompt the model without feature injection
        # Remove [s]v[e] tokens and replace with a placeholder
        question_baseline = re.sub(r'\[s\]v\[e\]', 'feature', question)
        
        # Tokenize and generate
        try:
            enc = tokenizer(question_baseline, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
            
            prompt_len = input_ids.shape[1]
            generated_ids = outputs[0, prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Parse prediction
            pred_result = parse_explanation(generated_text)
            
            if pred_result is None:
                pred_result = PredictionResult(
                    has_changed=False,
                    next_token="",
                    explanation=generated_text,
                    raw_output=generated_text
                )
        except Exception as e:
            if verbose:
                print(f"Error generating explanation: {e}")
            pred_result = PredictionResult(
                has_changed=False,
                next_token="",
                explanation="",
                raw_output=""
            )
        
        # Get ground truth
        gt_has_changed = item["has_changed"]
        gt_token = item["patched_next_token"] if gt_has_changed else item["original_next_token"]
        gt_explanation = item["answer"]
        
        predictions.append({
            "has_changed": pred_result.has_changed,
            "next_token": pred_result.next_token,
            "explanation": pred_result.explanation,
            "raw_output": pred_result.raw_output
        })
        
        ground_truth.append({
            "has_changed": gt_has_changed,
            "next_token": gt_token,
            "explanation": gt_explanation
        })
    
    # Compute metrics
    pred_has_changed = [p["has_changed"] for p in predictions]
    gt_has_changed = [g["has_changed"] for g in ground_truth]
    
    pred_tokens = [p["next_token"] for p in predictions]
    gt_tokens = [g["next_token"] for g in ground_truth]
    
    pred_explanations = [p["explanation"] for p in predictions]
    gt_explanations = [g["explanation"] for g in ground_truth]
    
    exact_match = compute_exact_match(pred_explanations, gt_explanations)
    has_changed_f1 = compute_has_changed_f1(pred_has_changed, gt_has_changed)
    content_match = compute_content_match(pred_tokens, gt_tokens)
    
    return {
        "exact_match": exact_match,
        "has_changed_f1": has_changed_f1,
        "content_match": content_match,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "num_samples": len(test_data)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate activation patching explainer models")
    parser.add_argument("--explainer_model", type=str, default=None,
                       help="Path to trained explainer model checkpoint (or pretrained model for baseline)")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test dataset JSON file")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save evaluation results JSON")
    parser.add_argument("--num_runs", type=int, default=5,
                       help="Number of evaluation runs for statistical significance")
    parser.add_argument("--bootstrap_samples", type=int, default=1000,
                       help="Number of bootstrap samples per run (0 to disable)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--run_ablations", action="store_true",
                       help="Run ablation studies (remove activation, layer, token)")
    parser.add_argument("--baseline_only", action="store_true",
                       help="Evaluate untrained baseline model (no fine-tuning)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")
    
    args = parser.parse_args()
    
    if args.explainer_model is None:
        parser.error("--explainer_model is required")
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Load model
    print(f"Loading explainer model from {args.explainer_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.explainer_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.explainer_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Ensure special tokens exist
    special_tokens = ["[s]", "[e]"]
    existing_vocab = set(tokenizer.get_vocab().keys())
    tokens_to_add = [tok for tok in special_tokens if tok not in existing_vocab]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added special tokens: {tokens_to_add}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Model loaded successfully")
    
    # Run evaluation
    all_results = {}
    
    if args.baseline_only:
        # Evaluate untrained baseline
        print("\nEvaluating untrained baseline model...")
        baseline_results = run_evaluation_with_bootstrap(
            model, tokenizer, test_data,
            ablation_config=None,
            num_runs=args.num_runs,
            bootstrap_samples=args.bootstrap_samples,
            verbose=args.verbose,
            use_baseline=True,
            max_new_tokens=args.max_new_tokens
        )
        all_results["baseline"] = baseline_results
    else:
        # Full model (no ablation)
        print("\nEvaluating full model...")
        full_results = run_evaluation_with_bootstrap(
            model, tokenizer, test_data,
            ablation_config=None,
            num_runs=args.num_runs,
            bootstrap_samples=args.bootstrap_samples,
            verbose=args.verbose,
            max_new_tokens=args.max_new_tokens
        )
        all_results["full"] = full_results
    
    # Ablation studies
    if args.run_ablations:
        ablation_configs = [
            ("activation", {"remove_activation": True, "remove_layer": False, "remove_token": False}),
            ("layer", {"remove_activation": False, "remove_layer": True, "remove_token": False}),
            ("token", {"remove_activation": False, "remove_layer": False, "remove_token": True}),
        ]
        
        for ablation_name, ablation_config in ablation_configs:
            print(f"\nEvaluating with ablation: {ablation_name}...")
            ablation_results = run_evaluation_with_bootstrap(
                model, tokenizer, test_data,
                ablation_config=ablation_config,
                num_runs=args.num_runs,
                bootstrap_samples=args.bootstrap_samples,
                verbose=args.verbose,
                max_new_tokens=args.max_new_tokens
            )
            all_results[ablation_name] = ablation_results
    
    # Print results
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(format_results_table(all_results, model_name=args.explainer_model))
    print("=" * 65)
    
    # Save results
    if args.output_file:
        # Convert to JSON-serializable format
        output_data = {
            "model": args.explainer_model,
            "test_data": args.test_data,
            "num_runs": args.num_runs,
            "results": {
                config: {
                    metric: {"mean": mean, "std_error": std_error}
                    for metric, (mean, std_error) in metrics.items()
                }
                for config, metrics in all_results.items()
            }
        }
        
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

