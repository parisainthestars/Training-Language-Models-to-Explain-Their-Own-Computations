"""
Inference script for activation patching explainer models.

This script demonstrates how trained and baseline models generate explanations
for activation patching questions. It allows interactive inference and comparison
between trained and untrained models.

Usage:
    # Run inference on dataset examples
    python inference.py \
        --model ./explainer_checkpoints \
        --dataset activation_patch_dataset.json \
        --num_examples 10
    
    # Compare trained vs baseline
    python inference.py \
        --model ./explainer_checkpoints \
        --baseline_model Qwen/Qwen3-8B \
        --dataset activation_patch_dataset.json \
        --num_examples 5
    
    # Interactive mode with custom input
    python inference.py \
        --model ./explainer_checkpoints \
        --interactive
"""

import json
import re
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import argparse
from tqdm import tqdm
import random

# Import from activation_patching.py
from activation_patching import find_v_index


@dataclass
class InferenceResult:
    """Result of a single inference."""
    question: str
    generated_explanation: str
    ground_truth: Optional[str] = None
    has_changed_pred: Optional[bool] = None
    has_changed_gt: Optional[bool] = None
    next_token_pred: Optional[str] = None
    next_token_gt: Optional[str] = None
    is_correct: Optional[bool] = None
    model_type: str = "trained"  # "trained" or "baseline"


def parse_explanation(explanation_text: str) -> Tuple[Optional[bool], Optional[str]]:
    """
    Parse explanation text to extract has_changed and next_token.
    
    Returns:
        (has_changed, next_token) or (None, None) if parsing fails
    """
    explanation_text = explanation_text.strip()
    
    # Pattern for "change to"
    change_pattern = r"the output would change to\s*<<<([^>]+)>>>\.?"
    match = re.search(change_pattern, explanation_text, re.IGNORECASE)
    if match:
        token = match.group(1).strip()
        return True, token
    
    # Pattern for "remain unchanged"
    unchanged_pattern = r"the output would remain unchanged from\s*<<<([^>]+)>>>\.?"
    match = re.search(unchanged_pattern, explanation_text, re.IGNORECASE)
    if match:
        token = match.group(1).strip()
        return False, token
    
    # Fallback: try to extract any token in <<<...>>>
    token_pattern = r"<<<([^>]+)>>>"
    match = re.search(token_pattern, explanation_text, re.IGNORECASE)
    if match:
        token = match.group(1).strip()
        has_changed = "change" in explanation_text.lower() and "unchanged" not in explanation_text.lower()
        return has_changed, token
    
    return None, None


def generate_explanation_with_feature(
    model,
    tokenizer,
    question: str,
    feature_vector: List[float],
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    do_sample: bool = False,
    debug: bool = False
) -> str:
    """
    Generate explanation using trained model with feature vector injection.
    
    Args:
        model: Trained explainer model
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
        if debug:
            print(f"DEBUG: Found v_idx at position {v_idx}")
    except Exception as e:
        if debug:
            print(f"DEBUG: Error finding v_idx: {e}")
        # Fallback: find [s] token and use next position
        s_id = tokenizer.convert_tokens_to_ids("[s]")
        s_positions = (input_ids[0] == s_id).nonzero(as_tuple=True)[0]
        if len(s_positions) > 0:
            v_idx = int(s_positions[0].item()) + 1
            if debug:
                print(f"DEBUG: Using fallback v_idx at position {v_idx} (after [s])")
        else:
            v_idx = len(input_ids[0]) // 2
            if debug:
                print(f"DEBUG: Using fallback v_idx at position {v_idx} (middle of sequence)")
    
    # Prepare feature vector
    v = torch.tensor(feature_vector, dtype=torch.float32)
    v_norm = v / (v.norm() + 1e-8)  # Normalize
    if debug:
        print(f"DEBUG: Feature vector shape: {v.shape}, norm: {v_norm.norm().item():.4f}")
    
    # Get embeddings and inject v
    emb_layer = model.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids)
    if debug:
        print(f"DEBUG: Input embeddings shape: {inputs_embeds.shape}")
    
    # Inject v at v_idx
    if v_idx < inputs_embeds.size(1):
        inputs_embeds[0, v_idx, :] = v_norm.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
        if debug:
            print(f"DEBUG: Injected feature vector at position {v_idx}")
    else:
        if debug:
            print(f"DEBUG: WARNING - v_idx {v_idx} >= sequence length {inputs_embeds.size(1)}")
    
    # Generate with better stopping criteria
    with torch.no_grad():
        # Create stopping criteria - stop at period after <<<...>>> pattern
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id,
            eos_token_id=eos_token_id,
            # Stop generation early if we see the pattern
            stopping_criteria=None,
        )
    
    # Decode only the generated part
    prompt_len = input_ids.shape[1]
    generated_ids = outputs[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    if debug:
        print(f"DEBUG: Generated {len(generated_ids)} tokens")
        print(f"DEBUG: Generated IDs: {generated_ids.tolist()[:10]}...")  # First 10 tokens
        print(f"DEBUG: Generated text (first 100 chars): {generated_text[:100]}")
    
    # If empty, try decoding the full output to debug
    if not generated_text:
        if debug:
            print("DEBUG: Generated text is empty, trying full decode...")
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Try to extract just the continuation
        if len(full_text) > len(question):
            generated_text = full_text[len(question):].strip()
            if debug:
                print(f"DEBUG: Extracted from full text: {generated_text[:100]}")
    
    return generated_text


def generate_explanation_baseline(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    do_sample: bool = False
) -> str:
    """
    Generate explanation using baseline (untrained) model without feature injection.
    
    Args:
        model: Untrained baseline model
        tokenizer: Tokenizer
        question: Input question (will remove [s]v[e] tokens)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
    
    Returns:
        Generated explanation text
    """
    # Remove [s]v[e] tokens for baseline (no feature injection)
    question_baseline = re.sub(r'\[s\]v\[e\]', 'feature', question)
    
    # Tokenize and generate
    enc = tokenizer(question_baseline, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    
    with torch.no_grad():
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id,
            eos_token_id=eos_token_id,
        )
    
    # Decode only the generated part
    prompt_len = input_ids.shape[1]
    generated_ids = outputs[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # If empty, try decoding the full output to debug
    if not generated_text:
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Try to extract just the continuation
        if len(full_text) > len(question_baseline):
            generated_text = full_text[len(question_baseline):].strip()
    
    return generated_text


def load_model(model_path: str, is_baseline: bool = False):
    """
    Load model and tokenizer.
    
    Args:
        model_path: Path to model checkpoint or HuggingFace model name
        is_baseline: Whether this is a baseline (untrained) model
    
    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading {'baseline' if is_baseline else 'trained'} model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Ensure special tokens exist (only for trained models)
    if not is_baseline:
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
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    question: str,
    feature_vector: Optional[List[float]] = None,
    ground_truth: Optional[str] = None,
    has_changed_gt: Optional[bool] = None,
    next_token_gt: Optional[str] = None,
    use_baseline: bool = False,
    max_new_tokens: int = 50,
    debug: bool = False
) -> InferenceResult:
    """
    Run inference on a single example.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        question: Input question
        feature_vector: Feature vector (required for trained models)
        ground_truth: Ground truth explanation
        has_changed_gt: Ground truth has_changed value
        next_token_gt: Ground truth next token
        use_baseline: Whether to use baseline mode (no feature injection)
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        InferenceResult object
    """
    # Generate explanation
    if use_baseline:
        generated_text = generate_explanation_baseline(
            model, tokenizer, question, max_new_tokens=max_new_tokens
        )
    else:
        if feature_vector is None:
            raise ValueError("feature_vector is required for trained models")
        generated_text = generate_explanation_with_feature(
            model, tokenizer, question, feature_vector, 
            max_new_tokens=max_new_tokens, debug=debug
        )
    
    # Parse prediction
    has_changed_pred, next_token_pred = parse_explanation(generated_text)
    
    # Check correctness if ground truth available
    is_correct = None
    if ground_truth is not None:
        is_correct = generated_text.strip().lower() == ground_truth.strip().lower()
    elif has_changed_gt is not None and has_changed_pred is not None:
        is_correct = has_changed_gt == has_changed_pred
    
    return InferenceResult(
        question=question,
        generated_explanation=generated_text,
        ground_truth=ground_truth,
        has_changed_pred=has_changed_pred,
        has_changed_gt=has_changed_gt,
        next_token_pred=next_token_pred,
        next_token_gt=next_token_gt,
        is_correct=is_correct,
        model_type="baseline" if use_baseline else "trained"
    )


def print_inference_result(result: InferenceResult, show_details: bool = True):
    """
    Print inference result in a formatted way.
    
    Args:
        result: InferenceResult to print
        show_details: Whether to show detailed information
    """
    print("\n" + "=" * 80)
    print(f"MODEL: {result.model_type.upper()}")
    print("=" * 80)
    
    if show_details:
        print("\n📝 QUESTION:")
        print("-" * 80)
        # Truncate long questions
        question_display = result.question
        if len(question_display) > 500:
            question_display = question_display[:500] + "..."
        print(question_display)
    
    print("\n🤖 GENERATED EXPLANATION:")
    print("-" * 80)
    if result.generated_explanation:
        print(result.generated_explanation)
    else:
        print("[EMPTY OUTPUT - Model generated no text]")
    
    if result.ground_truth is not None:
        print("\n✅ GROUND TRUTH:")
        print("-" * 80)
        print(result.ground_truth)
    
    if show_details:
        print("\n📊 PREDICTION DETAILS:")
        print("-" * 80)
        if result.has_changed_pred is not None:
            status = "CHANGED" if result.has_changed_pred else "UNCHANGED"
            print(f"  Predicted: Output would {status}")
            if result.next_token_pred:
                print(f"  Predicted token: {result.next_token_pred}")
        else:
            print("  Predicted: [Could not parse explanation]")
        
        if result.has_changed_gt is not None:
            status = "CHANGED" if result.has_changed_gt else "UNCHANGED"
            print(f"  Ground truth: Output would {status}")
            if result.next_token_gt:
                print(f"  Ground truth token: {result.next_token_gt}")
        
        if result.is_correct is not None:
            status = "✓ CORRECT" if result.is_correct else "✗ INCORRECT"
            print(f"  Match: {status}")
        else:
            print("  Match: [Cannot determine - parsing failed]")
    
    print("=" * 80)


def compare_models(
    trained_model,
    trained_tokenizer,
    baseline_model,
    baseline_tokenizer,
    question: str,
    feature_vector: List[float],
    ground_truth: Optional[str] = None,
    has_changed_gt: Optional[bool] = None,
    next_token_gt: Optional[str] = None,
    debug: bool = False
):
    """
    Compare trained and baseline models on the same example.
    
    Args:
        trained_model: Trained explainer model
        trained_tokenizer: Tokenizer for trained model
        baseline_model: Baseline (untrained) model
        baseline_tokenizer: Tokenizer for baseline model
        question: Input question
        feature_vector: Feature vector
        ground_truth: Ground truth explanation
        has_changed_gt: Ground truth has_changed value
        next_token_gt: Ground truth next token
    """
    print("\n" + "=" * 80)
    print("COMPARISON: TRAINED vs BASELINE")
    print("=" * 80)
    
    # Run trained model
    trained_result = run_inference(
        trained_model, trained_tokenizer, question, feature_vector,
        ground_truth, has_changed_gt, next_token_gt,
        use_baseline=False,
        debug=debug
    )
    
    # Run baseline model
    baseline_result = run_inference(
        baseline_model, baseline_tokenizer, question, feature_vector,
        ground_truth, has_changed_gt, next_token_gt,
        use_baseline=True,
        debug=debug
    )
    
    # Print comparison
    print("\n📝 QUESTION:")
    print("-" * 80)
    question_display = question
    if len(question_display) > 500:
        question_display = question_display[:500] + "..."
    print(question_display)
    
    print("\n🎯 TRAINED MODEL:")
    print("-" * 80)
    if trained_result.generated_explanation:
        print(trained_result.generated_explanation)
    else:
        print("[EMPTY OUTPUT - Model generated no text]")
    if trained_result.is_correct is not None:
        status = "✓ CORRECT" if trained_result.is_correct else "✗ INCORRECT"
        print(f"\nMatch: {status}")
    else:
        print("\nMatch: [Cannot determine - parsing failed]")
    
    print("\n🔵 BASELINE MODEL:")
    print("-" * 80)
    if baseline_result.generated_explanation:
        print(baseline_result.generated_explanation)
    else:
        print("[EMPTY OUTPUT - Model generated no text]")
    if baseline_result.is_correct is not None:
        status = "✓ CORRECT" if baseline_result.is_correct else "✗ INCORRECT"
        print(f"\nMatch: {status}")
    else:
        print("\nMatch: [Cannot determine - parsing failed]")
    
    if ground_truth is not None:
        print("\n✅ GROUND TRUTH:")
        print("-" * 80)
        print(ground_truth)
    
    print("=" * 80)


def interactive_mode(model, tokenizer, use_baseline: bool = False):
    """
    Run interactive inference mode.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        use_baseline: Whether using baseline model
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE INFERENCE MODE")
    print("=" * 80)
    print("Enter activation patching questions. Type 'quit' to exit.")
    print("For trained models, you'll need to provide a feature vector.")
    print("=" * 80)
    
    while True:
        print("\n" + "-" * 80)
        question = input("\nEnter question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        # For trained models, ask for feature vector
        feature_vector = None
        if not use_baseline:
            feature_input = input("Enter feature vector (comma-separated numbers, or 'skip'): ").strip()
            if feature_input.lower() != 'skip':
                try:
                    feature_vector = [float(x.strip()) for x in feature_input.split(',')]
                except ValueError:
                    print("Invalid feature vector format. Using random vector.")
                    feature_vector = np.random.randn(4096).tolist()  # Default size
        
        # Generate
        try:
            if use_baseline:
                generated = generate_explanation_baseline(model, tokenizer, question)
            else:
                if feature_vector is None:
                    print("Warning: No feature vector provided. Using random vector.")
                    feature_vector = np.random.randn(4096).tolist()
                generated = generate_explanation_with_feature(model, tokenizer, question, feature_vector)
            
            print("\n🤖 GENERATED EXPLANATION:")
            print("-" * 80)
            print(generated)
        except Exception as e:
            print(f"\nError generating explanation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on activation patching explainer models")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained explainer model checkpoint")
    parser.add_argument("--baseline_model", type=str, default=None,
                       help="Path to baseline (untrained) model for comparison")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to dataset JSON file for examples")
    parser.add_argument("--num_examples", type=int, default=10,
                       help="Number of examples to run inference on")
    parser.add_argument("--example_indices", type=int, nargs="+", default=None,
                       help="Specific example indices to run (overrides num_examples)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--compare", action="store_true",
                       help="Compare trained vs baseline models")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for example selection")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load trained model
    trained_model, trained_tokenizer = load_model(args.model, is_baseline=False)
    
    # Load baseline model if specified
    baseline_model = None
    baseline_tokenizer = None
    if args.baseline_model:
        baseline_model, baseline_tokenizer = load_model(args.baseline_model, is_baseline=True)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(trained_model, trained_tokenizer, use_baseline=False)
        return
    
    # Load dataset if provided
    if args.dataset:
        print(f"\nLoading dataset from {args.dataset}...")
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples")
        
        # Select examples
        if args.example_indices:
            examples = [dataset[i] for i in args.example_indices if i < len(dataset)]
        else:
            examples = random.sample(dataset, min(args.num_examples, len(dataset)))
        
        print(f"\nRunning inference on {len(examples)} examples...")
        
        # Run inference
        for idx, example in enumerate(tqdm(examples, desc="Inference")):
            question = example["question"]
            feature_vector = example["feature_vector"]
            ground_truth = example.get("answer")
            has_changed_gt = example.get("has_changed")
            next_token_gt = example.get("patched_next_token") if has_changed_gt else example.get("original_next_token")
            
            if args.compare and baseline_model is not None:
                # Compare mode
                compare_models(
                    trained_model, trained_tokenizer,
                    baseline_model, baseline_tokenizer,
                    question, feature_vector,
                    ground_truth, has_changed_gt, next_token_gt,
                    debug=args.debug
                )
            else:
                # Single model mode
                result = run_inference(
                    trained_model, trained_tokenizer, question, feature_vector,
                    ground_truth, has_changed_gt, next_token_gt,
                    use_baseline=False,
                    max_new_tokens=args.max_new_tokens,
                    debug=args.debug
                )
                print_inference_result(result, show_details=True)
                
                # Also run baseline if available
                if baseline_model is not None:
                    baseline_result = run_inference(
                        baseline_model, baseline_tokenizer, question, feature_vector,
                        ground_truth, has_changed_gt, next_token_gt,
                        use_baseline=True,
                        max_new_tokens=args.max_new_tokens,
                        debug=args.debug
                    )
                    print_inference_result(baseline_result, show_details=True)
    else:
        # No dataset provided, run interactive mode
        print("No dataset provided. Running in interactive mode...")
        interactive_mode(trained_model, trained_tokenizer, use_baseline=False)


if __name__ == "__main__":
    main()

