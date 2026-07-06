import json
import random
import torch
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
try:
    from sklearn.metrics import f1_score
except ImportError:
    f1_score = None
def get_distractor_options(entry, dataset, num_options=6):
    true_ans = entry["expected_factual"]
    counterfactual_ans = entry["expected_counterfactual"]

    relation_id = entry["relation_id"]
    # collect unique distractors, exclude current entry
    distractors = list({
        e["expected_factual"]
        for e in dataset
        if e["relation_id"] == relation_id
        and e["case_id"] != entry["case_id"]
        and e["expected_factual"] not in [true_ans, counterfactual_ans]
    })
    random.shuffle(distractors)
    return distractors[:num_options]

def build_prompt(x_entry, dataset, num_options=6):
    options = get_distractor_options(x_entry, dataset, num_options)
    # only include factual sentence + multi-choice instruction
    prompt_lines = [
        x_entry["prompt"].format(x_entry["subject"]),
        "Respond with one of " + " or ".join(options) + " and nothing else."
    ]
    return "\n".join(prompt_lines)

# -----------------------------
# Generate x and x' entries
# -----------------------------
def generate_x_xprime(entry):
    rr = entry["requested_rewrite"]
    subject = rr["subject"]
    prompt_template = rr["prompt"]

    x = {
        "prompt": prompt_template,
        "subject": subject,
        "expected_factual": rr["target_true"]["str"],
        "expected_counterfactual": rr["target_new"]["str"],
        "relation_id": rr["relation_id"],
        "case_id": entry["case_id"],
        "type": "factual"
    }

    x_prime = {
        "prompt": prompt_template,
        "subject": subject,
        "expected_factual": rr["target_true"]["str"],
        "expected_counterfactual": rr["target_new"]["str"],
        "relation_id": rr["relation_id"],
        "case_id": entry["case_id"],
        "type": "counterfactual"
    }

    return x, x_prime

def generate_all_x_xprime(dataset):
    xs, xprimes = [], []
    for entry in dataset:
        x, xp = generate_x_xprime(entry)
        xs.append(x)
        xprimes.append(xp)
    return xs, xprimes

# -----------------------------
# Token Type Classification
# -----------------------------
@dataclass
class TokenTypeInfo:
    token_type: str  # "Subject Final", "Relation", "Orig Answer Option", "New Answer Option", "Other Answer Option", "Question q"
    token_text: str
    token_position: int

def classify_token_types(
    tokenizer, 
    prompt_x: str, 
    prompt_xprime: str,
    subject_x: str,
    subject_xprime: str,
    expected_factual: str,
    expected_counterfactual: str,
    answer_options: List[str]
) -> List[TokenTypeInfo]:
    """
    Classify tokens into types:
    1. Subject Final: final token of subject
    2. Relation: relation tokens (e.g., "is the capital of")
    3. Orig Answer Option: answer option corresponding to x (e.g., Italy)
    4. New Answer Option: answer option corresponding to x' (e.g., France)
    5. Other Answer Option: other answer options
    6. Question q: all other tokens
    """
    # Tokenize both prompts
    tokens_x = tokenizer.tokenize(prompt_x)
    tokens_xprime = tokenizer.tokenize(prompt_xprime)
    
    # Find subject final token position
    subject_tokens_x = tokenizer.tokenize(subject_x)
    subject_final_pos = len(subject_tokens_x) - 1
    
    # Find relation tokens (between subject and answer options)
    # Relation is typically "is the capital of" or similar
    relation_start = len(subject_tokens_x)
    # Find where answer options start
    answer_start = prompt_x.find("Respond with one of")
    if answer_start == -1:
        answer_start = len(prompt_x)
    
    answer_tokens_x = tokenizer.tokenize(prompt_x[answer_start:])
    relation_end = len(tokens_x) - len(answer_tokens_x)
    
    token_types = []
    for i, token in enumerate(tokens_x):
        token_text = token
        
        # Check if it's subject final
        if i == subject_final_pos:
            token_types.append(TokenTypeInfo("Subject Final", token_text, i))
        # Check if it's relation
        elif relation_start <= i < relation_end:
            token_types.append(TokenTypeInfo("Relation", token_text, i))
        # Check if it's in answer options section
        elif i >= relation_end:
            # Check which answer option this token belongs to
            token_lower = token_text.lower().strip("Ġ")
            if expected_factual.lower() in token_lower or token_lower in expected_factual.lower():
                token_types.append(TokenTypeInfo("Orig Answer Option", token_text, i))
            elif expected_counterfactual.lower() in token_lower or token_lower in expected_counterfactual.lower():
                token_types.append(TokenTypeInfo("New Answer Option", token_text, i))
            elif any(opt.lower() in token_lower or token_lower in opt.lower() for opt in answer_options 
                     if opt not in [expected_factual, expected_counterfactual]):
                token_types.append(TokenTypeInfo("Other Answer Option", token_text, i))
            else:
                token_types.append(TokenTypeInfo("Question q", token_text, i))
        else:
            token_types.append(TokenTypeInfo("Question q", token_text, i))
    
    return token_types

# -----------------------------
# Layer Chunking
# -----------------------------
def get_layer_chunks(num_layers: int, model_type: str = "llama") -> List[List[int]]:
    """
    Divide layers into chunks for comprehensive search.
    Increments: 8 for Llama, 9 for Qwen
    """
    chunk_size = 8 if model_type.lower() in ["llama", "llama3"] else 9
    
    chunks = []
    for start in range(0, num_layers, chunk_size):
        end = min(start + chunk_size, num_layers)
        chunks.append(list(range(start, end)))
    
    return chunks

def divide_layers_into_blocks(num_layers: int, num_blocks: int = 4) -> List[List[int]]:
    """
    Divide layers into blocks for multi-layer patching.
    Used to compute aggregate representation v = avg_ℓk(h_ℓk,t(x'))
    """
    block_size = num_layers // num_blocks
    blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < num_blocks - 1 else num_layers
        blocks.append(list(range(start, end)))
    return blocks

# -----------------------------
# Activation Patching Functions
# -----------------------------
def get_hidden_states(model, tokenizer, prompt: str, layers: Optional[List[int]] = None):
    """
    Get hidden states for all layers or specified layers.
    Returns: dict[layer_idx] -> hidden_state tensor [seq_len, hidden_dim]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = {}
    all_hidden = outputs.hidden_states
    
    if layers is None:
        layers = list(range(len(all_hidden)))
    
    for layer in layers:
        if layer < len(all_hidden):
            hidden_states[layer] = all_hidden[layer][0].detach().clone()  # [seq_len, hidden_dim]
    
    return hidden_states, inputs["input_ids"].shape[1]

def compute_aggregate_representation(
    hidden_states_dict: Dict[int, torch.Tensor],
    token_position: int
) -> torch.Tensor:
    """
    Compute aggregate representation v = avg_ℓk(h_ℓk,t(x'))
    """
    representations = []
    for layer, hidden in hidden_states_dict.items():
        if token_position < hidden.shape[0]:
            representations.append(hidden[token_position, :])
    
    if len(representations) == 0:
        raise ValueError(f"No valid representations found for token position {token_position}")
    
    stacked = torch.stack(representations, dim=0)  # [num_layers, hidden_dim]
    return torch.mean(stacked, dim=0)  # [hidden_dim]

def forward_with_multi_layer_patch(
    model,
    tokenizer,
    prompt: str,
    aggregate_v: torch.Tensor,
    layers: List[int],
    token_position: int,
    max_new_tokens: int = 20
) -> Tuple[str, torch.Tensor]:
    """
    Forward pass with multi-layer patching.
    Inserts aggregate representation v at all specified layers.
    Returns: (generated_text, logits)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    
    if token_position >= seq_len:
        raise ValueError(f"Token position {token_position} >= sequence length {seq_len}")
    
    handles = []
    
    def create_hook(layer_idx):
        def hook(module, input, output):
                # Patch the activation at token_position
            if output.shape[1] > token_position:
                output[:, token_position, :] = aggregate_v.to(output.device)
            return output
        return hook
    
    # Register hooks for all specified layers
    for layer_idx in layers:
        if layer_idx < len(model.model.layers):
            handle = model.model.layers[layer_idx].register_forward_hook(create_hook(layer_idx))
            handles.append(handle)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()
    
    # Generate next token
    next_token_id = torch.argmax(logits).unsqueeze(0)
    generated_ids = torch.cat([inputs["input_ids"][0], next_token_id], dim=0)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Extract only the answer (remove prompt)
    answer_only = generated_text[len(prompt):].strip()
    
    return answer_only, logits

def measure_output_change(
    original_output: str,
    patched_output: str,
    original_logits: Optional[torch.Tensor] = None,
    patched_logits: Optional[torch.Tensor] = None
) -> Tuple[bool, float]:
    """
    Measure change in model output.
    Returns: (has_changed: bool, change_score: float)
    """
    # Check if output text changed
    text_changed = original_output.strip().lower() != patched_output.strip().lower()
    
    # If logits available, compute KL divergence or other metric
    change_score = 0.0
    if original_logits is not None and patched_logits is not None:
        # Compute KL divergence between distributions
        orig_probs = torch.softmax(original_logits, dim=-1)
        patched_probs = torch.softmax(patched_logits, dim=-1)
        kl_div = torch.sum(orig_probs * torch.log(orig_probs / (patched_probs + 1e-10) + 1e-10))
        change_score = kl_div.item()
    
    return text_changed or (change_score > 0.1), change_score

# -----------------------------
# Activation Patching Procedure T_patch
# -----------------------------
def T_patch(
    model,
    tokenizer,
    prompt_x: str,
    prompt_xprime: str,
    layer_block: List[int],
    token_position: int,
    max_new_tokens: int = 20
) -> Dict[str, Any]:
    """
    Perform activation patching: M(x; h_ℓ,t(x) ← h_ℓ,t(x'))
    
    Args:
        model: Target model
        tokenizer: Tokenizer
        prompt_x: Original input x
        prompt_xprime: Counterfactual input x'
        layer_block: List of layers to patch (block of layers)
        token_position: Token position to patch
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Dictionary with:
        - original_next_token: Next token from M(x) (single token string)
        - patched_next_token: Next token from M(x; h_ℓ,t(x) ← h_ℓ,t(x')) (single token string)
        - has_changed: bool
        - change_score: float
        - original_logits: logits from M(x)
        - patched_logits: logits from patched model
        - original_token_id: token ID from M(x)
        - patched_token_id: token ID from patched model
    """
    # Get original output M(x) - only next token
    inputs_x = tokenizer(prompt_x, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs_x = model(**inputs_x)
        original_logits = outputs_x.logits[0, -1, :]
        original_token_id = torch.argmax(original_logits).item()
        original_next_token = tokenizer.decode([original_token_id], skip_special_tokens=True).strip()
    
    # Get hidden states from x' at specified layers
    hidden_states_xprime, seq_len_xprime = get_hidden_states(
        model, tokenizer, prompt_xprime, layers=layer_block
    )
    
    # Compute aggregate representation v = avg_ℓk(h_ℓk,t(x'))
    aggregate_v = compute_aggregate_representation(hidden_states_xprime, token_position)
    
    # Forward pass with patching: M(x; h_ℓ,t(x) ← v) - only next token
    inputs_x_patch = tokenizer(prompt_x, return_tensors="pt").to(model.device)
    seq_len = inputs_x_patch["input_ids"].shape[1]
    
    if token_position >= seq_len:
        raise ValueError(f"Token position {token_position} >= sequence length {seq_len}")
    
    handles = []
    
    def create_hook(layer_idx):
        def hook(module, input, output):
            if output.shape[1] > token_position:
                output[:, token_position, :] = aggregate_v.to(output.device)
            return output
        return hook
    
    for layer_idx in layer_block:
        if layer_idx < len(model.model.layers):
            handle = model.model.layers[layer_idx].register_forward_hook(create_hook(layer_idx))
            handles.append(handle)
    
    try:
        with torch.no_grad():
            outputs_patched = model(**inputs_x_patch)
            patched_logits = outputs_patched.logits[0, -1, :]
            patched_token_id = torch.argmax(patched_logits).item()
            patched_next_token = tokenizer.decode([patched_token_id], skip_special_tokens=True).strip()
    finally:
        for handle in handles:
            handle.remove()

    # Measure change - compare single tokens
    has_changed = original_token_id != patched_token_id
    change_score = 0.0
    if original_logits is not None and patched_logits is not None:
        orig_probs = torch.softmax(original_logits, dim=-1)
        patched_probs = torch.softmax(patched_logits, dim=-1)
        kl_div = torch.sum(orig_probs * torch.log(orig_probs / (patched_probs + 1e-10) + 1e-10))
        change_score = kl_div.item()
    
    return {
        "original_next_token": original_next_token,
        "patched_next_token": patched_next_token,
        "original_token_id": original_token_id,
        "patched_token_id": patched_token_id,
        "has_changed": has_changed,
        "change_score": change_score,
        "original_logits": original_logits.cpu(),
        "patched_logits": patched_logits.cpu(),
        "aggregate_v": aggregate_v.cpu()
    }

# -----------------------------
# Dataset Generation and Balancing
# -----------------------------
def generate_activation_patching_dataset(
    model,
    tokenizer,
    xs: List[Dict],
    xprimes: List[Dict],
    dataset: List[Dict],
    model_type: str = "llama",
    max_samples_per_combination: int = 10
) -> List[Dict]:
    """
    Generate comprehensive activation patching dataset.
    Performs search over all layer chunks and token positions.
    Balances dataset to avoid spurious correlations.
    """
    num_layers = len(model.model.layers)
    layer_chunks = get_layer_chunks(num_layers, model_type)
    layer_blocks = divide_layers_into_blocks(num_layers, num_blocks=4)
    
    all_samples = []
    combination_counts = defaultdict(int)
    
    # Statistics for balancing
    stats = defaultdict(lambda: defaultdict(int))
    
    print(f"Generating activation patching dataset...")
    print(f"Model has {num_layers} layers")
    print(f"Layer chunks: {len(layer_chunks)}")
    print(f"Layer blocks: {len(layer_blocks)}")
    
    for idx, (x, xprime) in enumerate(zip(xs, xprimes)):
        if idx % 10 == 0:
            print(f"Processing pair {idx}/{len(xs)}")
        
        # Build prompts
        prompt_x = build_prompt(x, xs, num_options=6)
        prompt_xprime = build_prompt(xprime, xs, num_options=6)
        
        # Classify token types
        token_types = classify_token_types(
            tokenizer,
            prompt_x,
            prompt_xprime,
            x["subject"],
            xprime["subject"],
            x["expected_factual"],
            x["expected_counterfactual"],
            get_distractor_options(x, xs, num_options=6)
        )
        
        # Tokenize to get actual positions
        tokens_x = tokenizer.tokenize(prompt_x)
        num_tokens = len(tokens_x)
        
        # Search over layer blocks and token positions
        for block_idx, layer_block in enumerate(layer_blocks):
            for token_pos in range(num_tokens):
                # Get token type
                token_type = token_types[token_pos].token_type if token_pos < len(token_types) else "Question q"
                
                # Check if we've exceeded max samples for this combination
                key = (block_idx, token_type, token_pos)
                if combination_counts[key] >= max_samples_per_combination:
                    continue
                
                try:
                    # Perform activation patching
                    result = T_patch(
                        model, tokenizer, prompt_x, prompt_xprime,
                        layer_block, token_pos, max_new_tokens=20
                    )
                    
                    # Create sample
                    sample = {
                        "case_id": x["case_id"],
                        "prompt_x": prompt_x,
                        "prompt_xprime": prompt_xprime,
                        "layer_block": layer_block,
                        "block_idx": block_idx,
                        "token_position": token_pos,
                        "token_type": token_type,
                        "original_next_token": result["original_next_token"],
                        "patched_next_token": result["patched_next_token"],
                        "original_token_id": result["original_token_id"],
                        "patched_token_id": result["patched_token_id"],
                        "has_changed": result["has_changed"],
                        "change_score": result["change_score"],
                        "aggregate_v": result["aggregate_v"].numpy().tolist(),
                        "expected_factual": x["expected_factual"],
                        "expected_counterfactual": x["expected_counterfactual"]
                    }
                    
                    all_samples.append(sample)
                    combination_counts[key] += 1
                    stats[block_idx][token_type] += 1
                    stats[block_idx]["has_changed_" + str(result["has_changed"])] += 1
                    
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
    
    print(f"\nGenerated {len(all_samples)} samples")
    print("\nStatistics:")
    for block_idx in sorted(stats.keys()):
        print(f"Block {block_idx}: {dict(stats[block_idx])}")
    
    return all_samples

def balance_dataset(samples: List[Dict]) -> List[Dict]:
    """
    Balance dataset to ensure roughly equal representation across has_changed categories
    and avoid over-representation of specific (token, layer) combinations.
    """
    # Group by (block_idx, token_type, has_changed)
    groups = defaultdict(list)
    for sample in samples:
        key = (sample["block_idx"], sample["token_type"], sample["has_changed"])
        groups[key].append(sample)
    
    # Find minimum count per group
    min_count = min(len(group) for group in groups.values()) if groups else 0
    target_count = max(min_count, 1)  # At least 1 per group
    
    balanced_samples = []
    for group_samples in groups.values():
        # Randomly sample to target_count
        if len(group_samples) > target_count:
            selected = random.sample(group_samples, target_count)
        else:
            selected = group_samples
        balanced_samples.extend(selected)
    
    random.shuffle(balanced_samples)
    return balanced_samples

# -----------------------------
# Training Data Preparation for Explainer
# -----------------------------
def prepare_explainer_training_data(samples: List[Dict], layer_idx: Optional[int] = None) -> List[Dict]:
    """
    Prepare training data for explainer model following the paper format.
    
    Prompt format:
    "If feature [s]v[e] at layer ℓ is inserted into token xt when processing the text <<<x>>>, 
     how would the output change?
     Respond with exactly one of the two options below, and nothing else:
     The output would remain unchanged from <<<X>>>.
     The output would change to <<<X>>>."
    
    Where X is a single next token.
    
    Args:
        samples: List of activation patching samples
        layer_idx: If provided, use a single layer index instead of layer_block for the prompt
    """
    training_data = []
    
    for sample in samples:
        # Use single layer index if provided, otherwise use first layer of block
        layer_for_prompt = layer_idx if layer_idx is not None else sample['layer_block'][0]
        
        # Format question following paper format with [s]v[e] tokens
        # The [s]v[e] marks where the continuous vector v will be inserted
        question = (
            f"If feature [s]v[e] at layer {layer_for_prompt} is inserted into token {sample['token_position']} "
            f"when processing the text <<<{sample['prompt_x']}>>>, how would the output change?\n\n"
            f"Respond with exactly one of the two options below, and nothing else:\n"
            f"The output would remain unchanged from <<<X>>>.\n"
            f"The output would change to <<<X>>>."
        )
        
        # Format answer - X is the single next token
        if sample["has_changed"]:
            answer = f"The output would change to <<<{sample['patched_next_token']}>>>."
        else:
            answer = f"The output would remain unchanged from <<<{sample['original_next_token']}>>>."
        
        training_item = {
            "input_text": sample["prompt_x"],
            "token_position": sample["token_position"],
            "layer_block": sample["layer_block"],
            "layer_for_prompt": layer_for_prompt,
            "block_idx": sample["block_idx"],
            "token_type": sample["token_type"],
            "feature_vector": sample["aggregate_v"],  # v to be inserted in embedding layer
            "question": question,
            "answer": answer,
            "has_changed": sample["has_changed"],
            "original_next_token": sample["original_next_token"],
            "patched_next_token": sample["patched_next_token"],
            "original_token_id": sample["original_token_id"],
            "patched_token_id": sample["patched_token_id"],
            "change_score": sample["change_score"]
        }
        
        training_data.append(training_item)
    
    return training_data

# -----------------------------
# Dataset Analysis and Statistics
# -----------------------------
def analyze_dataset_statistics(samples: List[Dict]) -> Dict[str, Any]:
    """
    Analyze activation patching dataset statistics.
    
    Returns comprehensive statistics about the dataset including:
    - Distribution across has_changed categories
    - Distribution across token types
    - Distribution across layer blocks
    - Distribution across (block, token_type, has_changed) combinations
    """
    stats = {
        "total_samples": len(samples),
        "has_changed": {
            "true": sum(1 for s in samples if s["has_changed"]),
            "false": sum(1 for s in samples if not s["has_changed"]),
        },
        "by_token_type": Counter(s["token_type"] for s in samples),
        "by_block": Counter(s["block_idx"] for s in samples),
        "by_combination": defaultdict(int),
        "min_samples_per_combination": float('inf'),
        "max_samples_per_combination": 0,
    }
    
    # Count by (block_idx, token_type, has_changed) combination
    for sample in samples:
        key = (sample["block_idx"], sample["token_type"], sample["has_changed"])
        stats["by_combination"][key] += 1
    
    if stats["by_combination"]:
        stats["min_samples_per_combination"] = min(stats["by_combination"].values())
        stats["max_samples_per_combination"] = max(stats["by_combination"].values())
    
    return stats

def print_dataset_statistics(stats: Dict[str, Any]):
    """Print formatted dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"\nTotal samples: {stats['total_samples']}")
    
    print(f"\nHas Changed Distribution:")
    true_count = stats["has_changed"]["true"]
    false_count = stats["has_changed"]["false"]
    total = stats["total_samples"]
    if total > 0:
        print(f"  Changed: {true_count} ({100*true_count/total:.1f}%)")
        print(f"  Unchanged: {false_count} ({100*false_count/total:.1f}%)")
    
    print(f"\nToken Type Distribution:")
    for token_type, count in stats["by_token_type"].most_common():
        if total > 0:
            print(f"  {token_type}: {count} ({100*count/total:.1f}%)")
        else:
            print(f"  {token_type}: {count}")
    
    print(f"\nLayer Block Distribution:")
    for block_idx, count in sorted(stats["by_block"].items()):
        if total > 0:
            print(f"  Block {block_idx}: {count} ({100*count/total:.1f}%)")
        else:
            print(f"  Block {block_idx}: {count}")
    
    print(f"\nCombination Statistics:")
    print(f"  Unique combinations: {len(stats['by_combination'])}")
    if stats['min_samples_per_combination'] != float('inf'):
        print(f"  Min samples per combination: {stats['min_samples_per_combination']}")
        print(f"  Max samples per combination: {stats['max_samples_per_combination']}")
    
    # Show some combination examples
    if stats["by_combination"]:
        print(f"\nTop 10 Combinations (block, token_type, has_changed):")
        for (block, token_type, has_changed), count in sorted(
            stats["by_combination"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]:
            print(f"  Block {block}, {token_type}, changed={has_changed}: {count}")
    
    print("="*60 + "\n")

def analyze_existing_dataset(dataset_file: str):
    """
    Analyze an existing activation patching dataset file.
    
    Args:
        dataset_file: Path to the dataset JSON file
    """
    print(f"\nAnalyzing dataset: {dataset_file}")
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {dataset_file} not found!")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        return
    
    if not data:
        print("Error: Dataset is empty!")
        return
    
    # Convert training data back to sample format if needed
    samples = []
    if "has_changed" in data[0] and "block_idx" in data[0]:
        # Already in sample format
        samples = data
    elif "question" in data[0] and "answer" in data[0]:
        # Training data format, convert back
        print("Converting training data format to sample format...")
        for item in data:
            sample = {
                "block_idx": item.get("block_idx", 0),
                "token_type": item.get("token_type", "Question q"),
                "token_position": item.get("token_position", 0),
                "has_changed": item.get("has_changed", False),
                "original_next_token": item.get("original_next_token", ""),
                "patched_next_token": item.get("patched_next_token", ""),
            }
            samples.append(sample)
    else:
        print("Warning: Unknown data format, attempting to use as-is")
        samples = data
    
    if not samples:
        print("Error: No valid samples found!")
        return
    
    # Analyze
    stats = analyze_dataset_statistics(samples)
    print_dataset_statistics(stats)
    
    # Check balance
    balance_check = check_dataset_balance(samples, min_samples_per_combination=5)
    print(f"\nBalance Check:")
    print(f"  Is balanced: {balance_check['is_balanced']}")
    print(f"  Missing combinations: {len(balance_check['missing_combinations'])}")
    print(f"  Under-represented: {len(balance_check['under_represented'])}")
    if balance_check['missing_combinations']:
        print(f"\n  Missing combinations (showing first 10):")
        for block, token_type, has_changed in balance_check['missing_combinations'][:10]:
            print(f"    Block {block}, {token_type}, changed={has_changed}")
    if balance_check['under_represented']:
        print(f"\n  Under-represented combinations (showing first 10):")
        for (block, token_type, has_changed), count in balance_check['under_represented'][:10]:
            print(f"    Block {block}, {token_type}, changed={has_changed}: {count} samples")

def check_dataset_balance(samples: List[Dict], min_samples_per_combination: int = 5) -> Dict[str, Any]:
    """
    Check if dataset is balanced and identify gaps.
    
    Returns:
        Dictionary with:
        - is_balanced: bool
        - missing_combinations: list of (block, token_type, has_changed) that need more samples
        - under_represented: combinations with fewer than min_samples_per_combination
    """
    # Group by combination
    combinations = defaultdict(list)
    for sample in samples:
        key = (sample["block_idx"], sample["token_type"], sample["has_changed"])
        combinations[key].append(sample)
    
    # Find missing or under-represented combinations
    missing = []
    under_represented = []
    
    # Get all possible combinations from existing data
    all_blocks = set(s["block_idx"] for s in samples)
    all_token_types = set(s["token_type"] for s in samples)
    all_has_changed = [True, False]
    
    for block in all_blocks:
        for token_type in all_token_types:
            for has_changed in all_has_changed:
                key = (block, token_type, has_changed)
                count = len(combinations[key])
                if count == 0:
                    missing.append(key)
                elif count < min_samples_per_combination:
                    under_represented.append((key, count))
    
    return {
        "is_balanced": len(missing) == 0 and len(under_represented) == 0,
        "missing_combinations": missing,
        "under_represented": under_represented,
        "total_combinations": len(combinations),
        "expected_combinations": len(all_blocks) * len(all_token_types) * len(all_has_changed),
    }

# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate activation patching dataset")
    parser.add_argument("--dataset_file", type=str, default="multi_counterfact.json",
                       help="Path to CounterFact dataset JSON file")
    parser.add_argument("--output_file", type=str, default="activation_patch_dataset.json",
                       help="Output file for generated dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Target model name")
    parser.add_argument("--max_samples_per_combination", type=int, default=10,
                       help="Max samples per (block, token_type, position) combination")
    parser.add_argument("--use_layer_chunks", action="store_true",
                       help="Also search over layer chunks (increments of 8/9)")
    parser.add_argument("--num_blocks", type=int, default=4,
                       help="Number of blocks to divide layers into")
    parser.add_argument("--check_existing", type=str, default=None,
                       help="Check statistics of existing dataset file")
    parser.add_argument("--max_pairs", type=int, default=None,
                       help="Maximum number of counterfactual pairs to process")
    
    args = parser.parse_args()
    
    # Check existing dataset if requested
    if args.check_existing:
        analyze_existing_dataset(args.check_existing)
        return
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_file}...")
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    xs, xprimes = generate_all_x_xprime(dataset)
    
    # Limit pairs if specified
    if args.max_pairs:
        xs = xs[:args.max_pairs]
        xprimes = xprimes[:args.max_pairs]
        print(f"Processing {len(xs)} counterfactual pairs (limited from {len(dataset)})")
    else:
        print(f"Processing {len(xs)} counterfactual pairs")

    # Load model
    model_name = args.model_name
    model_type = "qwen" if "qwen" in model_name.lower() else "llama"
    
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"Loaded model: {model_name}")
    print(f"Model type: {model_type}")
    print(f"Model has {len(model.model.layers)} layers")
    
    # Generate activation patching dataset
    print(f"\nGenerating activation patching dataset...")
    print(f"  Max samples per combination: {args.max_samples_per_combination}")
    print(f"  Use layer chunks: {args.use_layer_chunks}")
    print(f"  Number of blocks: {args.num_blocks}")
    
    samples = generate_activation_patching_dataset(
        model, tokenizer, xs, xprimes, dataset,
        model_type=model_type,
        max_samples_per_combination=args.max_samples_per_combination,
        use_layer_chunks=args.use_layer_chunks,
        num_blocks=args.num_blocks
    )
    
    print(f"\nGenerated {len(samples)} raw samples")
    
    # Balance dataset
    print("Balancing dataset...")
    balanced_samples = balance_dataset(samples)
    print(f"Balanced dataset: {len(balanced_samples)} samples")
    
    # Prepare training data for explainer
    print("Preparing training data...")
    training_data = prepare_explainer_training_data(balanced_samples)
    
    # Save dataset
    output_file = args.output_file
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(training_data)} training samples to {output_file}")
    
    # Print detailed statistics
    # Print comprehensive statistics
    print(f"\nFinal Dataset Statistics:")
    stats = analyze_dataset_statistics(balanced_samples)
    print_dataset_statistics(stats)
    
    # Check balance
    balance_check = check_dataset_balance(balanced_samples, min_samples_per_combination=5)
    print(f"\nBalance Check:")
    print(f"  Is balanced: {balance_check['is_balanced']}")
    print(f"  Missing combinations: {len(balance_check['missing_combinations'])}")
    print(f"  Under-represented: {len(balance_check['under_represented'])}")
    if balance_check['under_represented']:
        print(f"\n  Under-represented combinations (showing first 10):")
        for (block, token_type, has_changed), count in balance_check['under_represented'][:10]:
            print(f"    Block {block}, {token_type}, changed={has_changed}: {count} samples")
    
    block_counts = Counter(s["block_idx"] for s in balanced_samples)
    print(f"\nLayer block distribution:")
    for block_idx, count in block_counts.most_common():
        print(f"  Block {block_idx}: {count} samples")

# -----------------------------
# Evaluation Metrics
# -----------------------------
def compute_has_changed_f1(predictions: List[bool], ground_truth: List[bool]) -> float:
    """
    Compute Macro F1 score over "changed" and "unchanged" classes.
    
    Args:
        predictions: List of predicted has_changed values
        ground_truth: List of ground truth has_changed values
    
    Returns:
        Macro F1 score
    """
    if f1_score is None:
        raise ImportError("sklearn is required for F1 score computation. Install with: pip install scikit-learn")
    
    # Compute F1 for each class
    f1_changed = f1_score(ground_truth, predictions, pos_label=True, zero_division=0)
    f1_unchanged = f1_score(ground_truth, predictions, pos_label=False, zero_division=0)
    
    # Macro F1 is average of both
    macro_f1 = (f1_changed + f1_unchanged) / 2.0
    
    return macro_f1

def compute_content_match(predicted_tokens: List[str], ground_truth_tokens: List[str]) -> float:
    """
    Compute content match accuracy - whether predicted token matches ground truth.
    
    Args:
        predicted_tokens: List of predicted next tokens
        ground_truth_tokens: List of ground truth next tokens
    
    Returns:
        Content match accuracy (0-1)
    """
    matches = sum(1 for pred, gt in zip(predicted_tokens, ground_truth_tokens) 
                  if pred.strip().lower() == gt.strip().lower())
    return matches / len(predicted_tokens) if predicted_tokens else 0.0

def compute_exact_match(predicted_explanations: List[str], ground_truth_explanations: List[str]) -> float:
    """
    Compute exact match accuracy - full explanation must match exactly.
    
    Args:
        predicted_explanations: List of predicted full explanations
        ground_truth_explanations: List of ground truth full explanations
    
    Returns:
        Exact match accuracy (0-1)
    """
    matches = sum(1 for pred, gt in zip(predicted_explanations, ground_truth_explanations)
                  if pred.strip() == gt.strip())
    return matches / len(predicted_explanations) if predicted_explanations else 0.0

def evaluate_explainer_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate explainer predictions using all three metrics.
    
    Args:
        predictions: List of prediction dicts with keys:
            - has_changed: bool
            - next_token: str (predicted token)
            - explanation: str (full explanation)
        ground_truth: List of ground truth dicts with same keys
    
    Returns:
        Dictionary with metrics: has_changed_f1, content_match, exact_match
    """
    pred_has_changed = [p["has_changed"] for p in predictions]
    gt_has_changed = [g["has_changed"] for g in ground_truth]
    
    pred_tokens = [p.get("next_token", "") for p in predictions]
    gt_tokens = [g.get("next_token", "") for g in ground_truth]
    
    pred_explanations = [p.get("explanation", "") for p in predictions]
    gt_explanations = [g.get("explanation", "") for g in ground_truth]
    
    has_changed_f1 = compute_has_changed_f1(pred_has_changed, gt_has_changed)
    content_match = compute_content_match(pred_tokens, gt_tokens)
    exact_match = compute_exact_match(pred_explanations, gt_explanations)
    
    return {
        "has_changed_f1": has_changed_f1,
        "content_match": content_match,
        "exact_match": exact_match
    }

# -----------------------------
# Training Explainer Model
# -----------------------------
def find_v_index(input_ids: List[int], tokenizer) -> int:
    """
    Find the index of 'v' token between [s] and [e] in the tokenized sequence.
    """
    # Find [s] and [e] token IDs
    # These should be added to tokenizer if not present
    try:
        s_id = tokenizer.convert_tokens_to_ids("[s]")
        e_id = tokenizer.convert_tokens_to_ids("[e]")
        v_id = tokenizer.convert_tokens_to_ids("v")
    except:
        # Fallback: try to find special tokens
        s_id = tokenizer.get_vocab().get("[s]", None)
        e_id = tokenizer.get_vocab().get("[e]", None)
        v_id = tokenizer.get_vocab().get("v", None)
    
    if s_id is None or e_id is None or v_id is None:
        # If tokens don't exist, we need to add them or use a different approach
        # For now, assume v appears between [s] and [e] in the prompt
        # We'll search for the pattern manually
        for i, token_id in enumerate(input_ids):
            token = tokenizer.decode([token_id])
            if "[s]" in token or token_id == s_id:
                # Look for 'v' after [s]
                for j in range(i + 1, len(input_ids)):
                    if tokenizer.decode([input_ids[j]]).strip() == "v" or input_ids[j] == v_id:
                        return j
                break
    
    # Find [s] position
    s_positions = [i for i, tid in enumerate(input_ids) if tid == s_id]
    if not s_positions:
        raise ValueError("Could not find [s] token in input_ids")
    s_idx = s_positions[0]
    
    # Find [e] after [s]
    e_positions = [i for i, tid in enumerate(input_ids) if tid == e_id and i > s_idx]
    if not e_positions:
        raise ValueError("Could not find [e] token after [s] in input_ids")
    e_idx = e_positions[0]
    
    # Find 'v' between [s] and [e]
    v_positions = [i for i in range(s_idx + 1, e_idx) if input_ids[i] == v_id]
    if len(v_positions) != 1:
        # If exact match fails, return the position right after [s]
        return s_idx + 1
    return v_positions[0]

class ActivationPatchingDataset(Dataset):
    """
    Dataset for training explainer model on activation patching data.
    
    Each item contains:
    - question: Prompt with [s]v[e] tokens
    - answer: Ground truth explanation
    - feature_vector: v to be inserted at [s]v[e] position
    """
    def __init__(
        self,
        training_data: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        super().__init__()
        self.data = training_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure [s], [e] tokens exist in tokenizer
        self._ensure_special_tokens()
    
    def _ensure_special_tokens(self):
        """Add [s] and [e] tokens if they don't exist."""
        special_tokens = ["[s]", "[e]"]
        existing_special = set(self.tokenizer.get_vocab().keys())
        
        tokens_to_add = [tok for tok in special_tokens if tok not in existing_special]
        if tokens_to_add:
            self.tokenizer.add_tokens(tokens_to_add)
            print(f"Added special tokens to tokenizer: {tokens_to_add}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]
        feature_vector = item["feature_vector"]  # list or numpy array
        
        # Tokenize question (prompt with [s]v[e])
        question_enc = self.tokenizer(
            question,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        question_ids = question_enc["input_ids"]
        
        # Find index of 'v' token between [s] and [e]
        try:
            v_idx = find_v_index(question_ids, self.tokenizer)
        except Exception as e:
            # Fallback: assume v is right after [s]
            s_id = self.tokenizer.convert_tokens_to_ids("[s]")
            s_positions = [i for i, tid in enumerate(question_ids) if tid == s_id]
            v_idx = s_positions[0] + 1 if s_positions else len(question_ids) // 2
        
        # Tokenize answer
        answer_text = answer + self.tokenizer.eos_token
        answer_enc = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        answer_ids = answer_enc["input_ids"]
        
        # Full sequence: question + answer
        input_ids = question_ids + answer_ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        attention_mask = [1] * len(input_ids)
        
        # Labels: -100 for prompt tokens, actual ids for answer
        labels = [-100] * len(question_ids) + answer_ids
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        
        # Ensure all have same length
        assert len(input_ids) == len(labels) == len(attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "v": feature_vector,
            "v_idx": v_idx,
        }

@dataclass
class ContinuousTokenCollator:
    """
    Collator that injects continuous feature vector v at the [s]v[e] position
    in the embedding layer.
    """
    tokenizer: Any
    model: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract features - these are already tokenized
        input_ids_list = [f["input_ids"] for f in features]
        attn_list = [f["attention_mask"] for f in features]
        labels_list = [f["labels"] for f in features]
        v_list = [f["v"] for f in features]
        v_idx_list = [int(f["v_idx"]) for f in features]
        
        # Get padding token ID
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        # Find max length
        max_len = max(len(ids) for ids in input_ids_list)
        
        # Pad input_ids and attention_mask manually
        input_ids = []
        attention_mask = []
        for ids, attn in zip(input_ids_list, attn_list):
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_len)
            attention_mask.append(attn + [0] * pad_len)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Pad labels manually
        labels = []
        for label_seq in labels_list:
            pad_len = max_len - len(label_seq)
            labels.append(label_seq + [-100] * pad_len)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Get embeddings and inject v at v_idx
        # Work on CPU to avoid CUDA pinning issues
        emb_layer = self.model.get_input_embeddings()
        
        # Get embedding weights and manually compute embeddings on CPU
        # This avoids moving tensors to CUDA during collation
        embedding_weight = emb_layer.weight.data.cpu()  # [vocab_size, hidden_dim]
        input_ids_cpu = input_ids.cpu()  # Ensure input_ids is on CPU
        inputs_embeds = embedding_weight[input_ids_cpu]  # [B, L, hidden_dim] on CPU
        
        # Get dtype from embedding weight
        dtype = embedding_weight.dtype
        
        for i, (v, v_idx) in enumerate(zip(v_list, v_idx_list)):
            # Handle v - convert to tensor if needed, ensure it's on CPU
            if isinstance(v, torch.Tensor):
                v = v.detach().clone().cpu().to(dtype=dtype)
            else:
                v = torch.tensor(v, dtype=dtype)
            
            # Normalize v
            v = v / (v.norm() + 1e-8)
            # Inject at v_idx (accounting for padding)
            if v_idx < inputs_embeds.size(1):
                inputs_embeds[i, v_idx, :] = v
        
        # Return CPU tensors - trainer will move them to device
        return {
            "inputs_embeds": inputs_embeds,  # CPU tensor
            "attention_mask": attention_mask.cpu(),  # CPU tensor
            "labels": labels,  # CPU tensor
        }

def train_explainer_model(
    model,
    tokenizer,
    training_data: List[Dict],
    output_dir: str = "./explainer_checkpoints",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    eval_data: Optional[List[Dict]] = None,
    save_steps: int = 500,
    logging_steps: int = 50,
):
    """
    Train explainer model on activation patching data.
    
    Loss: Lpatch = -E[log p_E(E = Tpatch(M, x, t, ℓ1···i, v) | x, t, ℓ1···i, v)]
    
    Args:
        model: Explainer model to train (should be initialized)
        tokenizer: Tokenizer for the model
        training_data: List of training samples from prepare_explainer_training_data
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        eval_data: Optional evaluation data
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
    """
    # Resize token embeddings if new tokens were added
    if len(tokenizer) > model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized token embeddings to {len(tokenizer)}")
    
    # Create datasets
    train_dataset = ActivationPatchingDataset(training_data, tokenizer, max_length)
    eval_dataset = ActivationPatchingDataset(eval_data, tokenizer, max_length) if eval_data else None
    
    # Create collator
    data_collator = ContinuousTokenCollator(tokenizer=tokenizer, model=model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=save_steps if eval_dataset else None,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 only if bf16 not supported
        bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if supported (better stability)
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_grad_norm=1.0,  # Gradient clipping
    )
    
    # Custom trainer that uses inputs_embeds instead of input_ids
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Custom loss computation using inputs_embeds.
            Tensors are moved to model device here.
            """
            labels = inputs.pop("labels")
            inputs_embeds = inputs.pop("inputs_embeds")
            attention_mask = inputs.pop("attention_mask")
            
            # Move to model device
            device = next(model.parameters()).device
            inputs_embeds = inputs_embeds.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            
            return (loss, outputs) if return_outputs else loss
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"Starting training with {len(train_dataset)} samples...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete! Model saved to {output_dir}")
    return trainer

def main_train_explainer():
    """
    Main function to train explainer model on activation patching data.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train explainer model for activation patching")
    parser.add_argument("--dataset_file", type=str, default="activation_patch_dataset.json",
                       help="Path to activation patching dataset JSON file")
    parser.add_argument("--explainer_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                       help="Explainer model to train (use Qwen/Qwen2.5-3.8B-Instruct for 3.8B model)")
    parser.add_argument("--output_dir", type=str, default="./explainer_activation_patching",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--eval_split", type=float, default=0.1, help="Fraction of data for evaluation")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from {args.dataset_file}...")
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    print(f"Loaded {len(training_data)} training samples")
    
    # Split into train/eval
    random.shuffle(training_data)
    split_idx = int(len(training_data) * (1 - args.eval_split))
    train_data = training_data[:split_idx]
    eval_data = training_data[split_idx:] if args.eval_split > 0 else None
    
    print(f"Train samples: {len(train_data)}")
    if eval_data:
        print(f"Eval samples: {len(eval_data)}")
    
    # Load explainer model
    print(f"Loading explainer model: {args.explainer_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.explainer_model, trust_remote_code=True)
    # Load in FP32 - trainer will handle mixed precision conversion
    # This avoids gradient scaler issues with pre-loaded FP16 models
    model = AutoModelForCausalLM.from_pretrained(
        args.explainer_model,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded. Starting training...")
    
    # Train
    trainer = train_explainer_model(
        model=model,
        tokenizer=tokenizer,
        training_data=train_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        eval_data=eval_data,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )
    
    print("Training complete!")
    return trainer

def main_generate_data():
    """
    Main function to generate activation patching dataset with options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate activation patching dataset")
    parser.add_argument("--dataset_file", type=str, default="multi_counterfact.json",
                       help="Path to CounterFact dataset JSON file")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-8B",
                       help="Target model to perform activation patching on")
    parser.add_argument("--output_file", type=str, default="activation_patch_dataset.json",
                       help="Output file for training data")
    parser.add_argument("--raw_output_file", type=str, default="activation_patch_dataset_raw.json",
                       help="Output file for raw (unbalanced) data")
    parser.add_argument("--max_samples_per_combination", type=int, default=10,
                       help="Maximum samples per (block, token_type, position) combination")
    parser.add_argument("--max_pairs", type=int, default=None,
                       help="Maximum number of counterfactual pairs to process (None = all)")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze existing dataset, don't generate new data")
    parser.add_argument("--existing_dataset", type=str, default=None,
                       help="Path to existing dataset to analyze")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Analyze existing dataset
        if args.existing_dataset is None:
            args.existing_dataset = args.output_file
        
        print(f"Analyzing existing dataset: {args.existing_dataset}")
        try:
            with open(args.existing_dataset, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {args.existing_dataset} not found!")
            return
        
        if not data:
            print("Error: Dataset is empty!")
            return
        
        # Convert training data back to sample format if needed
        if "has_changed" in data[0] and "layer_block" in data[0]:
            # Already in sample format
            samples = data
        else:
            # Assume it's training data format, extract samples
            samples = []
            for item in data:
                sample = {
                    "block_idx": item.get("block_idx", 0),
                    "token_type": item.get("token_type", "Question q"),
                    "token_position": item.get("token_position", 0),
                    "has_changed": item.get("has_changed", False),
                }
                samples.append(sample)
        
        stats = analyze_activation_patching_data(samples)
        print_data_statistics(stats)
        is_balanced = check_data_balance(samples, threshold=0.2)
        if is_balanced:
            print("✓ Dataset is well-balanced!")
        else:
            print("⚠️  Dataset may need further balancing")
        return
    
    # Generate new dataset
    print("="*80)
    print("GENERATING ACTIVATION PATCHING DATASET")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_file}...")
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    xs, xprimes = generate_all_x_xprime(dataset)
    
    if args.max_pairs:
        xs = xs[:args.max_pairs]
        xprimes = xprimes[:args.max_pairs]
        print(f"Limited to {args.max_pairs} counterfactual pairs")
    
    print(f"Total counterfactual pairs: {len(xs)}")
    
    # Load model
    model_type = "qwen" if "qwen" in args.target_model.lower() else "llama"
    print(f"\nLoading target model: {args.target_model} (type: {model_type})...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print(f"Model loaded. Starting data generation...")
    
    # Generate activation patching dataset
    samples = generate_activation_patching_dataset(
        model, tokenizer, xs, xprimes, dataset,
        model_type=model_type,
        max_samples_per_combination=args.max_samples_per_combination
    )
    
    # Analyze raw samples
    print("\n" + "="*80)
    print("RAW DATASET STATISTICS (before balancing)")
    print("="*80)
    raw_stats = analyze_activation_patching_data(samples)
    print_data_statistics(raw_stats)
    
    # Balance dataset
    balanced_samples = balance_dataset(samples)
    print(f"\nBalanced dataset: {len(balanced_samples)} samples (from {len(samples)} raw samples)")
    
    # Analyze balanced samples
    print("\n" + "="*80)
    print("BALANCED DATASET STATISTICS (after balancing)")
    print("="*80)
    balanced_stats = analyze_activation_patching_data(balanced_samples)
    print_data_statistics(balanced_stats)
    
    # Check balance
    is_balanced = check_data_balance(balanced_samples, threshold=0.2)
    if is_balanced:
        print("✓ Dataset is well-balanced!")
    else:
        print("⚠️  Dataset may need further balancing")
    
    # Prepare training data for explainer
    training_data = prepare_explainer_training_data(balanced_samples)
    
    # Save dataset
    print(f"\nSaving datasets...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(training_data)} training samples to {args.output_file}")
    
    # Save raw samples too for analysis
    with open(args.raw_output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(samples)} raw samples to {args.raw_output_file}")
    
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode
        sys.argv = sys.argv[1:]  # Remove "train" from args
        main_train_explainer()
    elif len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Data generation mode with options
        sys.argv = sys.argv[1:]  # Remove "generate" from args
        main_generate_data()
    elif len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # Analysis mode
        sys.argv = sys.argv[1:]  # Remove "analyze" from args
        main_generate_data()  # Will use analyze_only flag
    else:
        # Default: original main() for backward compatibility
        main()
