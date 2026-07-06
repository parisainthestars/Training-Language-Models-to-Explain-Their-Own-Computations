#!/usr/bin/env python3
"""
Script to generate more activation patching data for training.

This script extends the existing activation patching dataset by:
1. Using more counterfactual pairs from CounterFact dataset
2. Configurable search over layer chunks and token positions
3. Balancing dataset to avoid spurious correlations
4. Providing statistics and validation

Usage:
    python generate_more_activation_patching_data.py \
        --model_name "Qwen/Qwen3-8B" \
        --num_pairs 100 \
        --max_samples_per_combination 20 \
        --output_file "activation_patch_dataset_extended.json" \
        --existing_dataset "activation_patch_dataset.json" \
        --resume_from_existing
"""

import json
import argparse
import random
from collections import defaultdict, Counter
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from activation_patching import (
    generate_all_x_xprime,
    generate_activation_patching_dataset,
    balance_dataset,
    prepare_explainer_training_data,
    get_layer_chunks,
    divide_layers_into_blocks
)


def load_existing_dataset(filepath: str) -> List[Dict]:
    """Load existing activation patching dataset."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} existing samples from {filepath}")
        return data
    except FileNotFoundError:
        print(f"No existing dataset found at {filepath}, starting fresh")
        return []


def get_existing_case_ids(existing_data: List[Dict]) -> set:
    """Extract case IDs from existing data to avoid duplicates."""
    case_ids = set()
    for sample in existing_data:
        if "case_id" in sample:
            case_ids.add(sample["case_id"])
        # Also check in nested structures
        if "input_text" in sample and "case_id" in sample.get("metadata", {}):
            case_ids.add(sample["metadata"]["case_id"])
    return case_ids


def filter_new_pairs(xs: List[Dict], xprimes: List[Dict], existing_case_ids: set) -> tuple:
    """Filter out pairs that already exist in the dataset."""
    new_xs = []
    new_xprimes = []
    
    for x, xprime in zip(xs, xprimes):
        if x["case_id"] not in existing_case_ids:
            new_xs.append(x)
            new_xprimes.append(xprime)
    
    print(f"Filtered: {len(new_xs)} new pairs (out of {len(xs)} total)")
    return new_xs, new_xprimes


def print_dataset_statistics(samples: List[Dict], label: str = "Dataset"):
    """Print comprehensive statistics about the dataset."""
    print(f"\n{'='*60}")
    print(f"{label} Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")
    
    if not samples:
        print("No samples to analyze")
        return
    
    # Has changed statistics
    has_changed_count = sum(1 for s in samples if s.get("has_changed", False))
    print(f"\nHas changed: {has_changed_count} ({100*has_changed_count/len(samples):.1f}%)")
    print(f"No change: {len(samples) - has_changed_count} ({100*(len(samples)-has_changed_count)/len(samples):.1f}%)")
    
    # Token type distribution
    token_type_counts = Counter(s.get("token_type", "Unknown") for s in samples)
    print(f"\nToken type distribution:")
    for token_type, count in token_type_counts.most_common():
        print(f"  {token_type}: {count} ({100*count/len(samples):.1f}%)")
    
    # Block distribution
    block_counts = Counter(s.get("block_idx", -1) for s in samples)
    print(f"\nBlock distribution:")
    for block_idx, count in sorted(block_counts.items()):
        print(f"  Block {block_idx}: {count} ({100*count/len(samples):.1f}%)")
    
    # Change score statistics
    change_scores = [s.get("change_score", 0.0) for s in samples if isinstance(s.get("change_score"), (int, float))]
    if change_scores:
        print(f"\nChange score statistics:")
        print(f"  Mean: {sum(change_scores)/len(change_scores):.4f}")
        print(f"  Min: {min(change_scores):.4f}")
        print(f"  Max: {max(change_scores):.4f}")
    
    # Unique case IDs
    case_ids = set(s.get("case_id") for s in samples if s.get("case_id"))
    print(f"\nUnique case IDs: {len(case_ids)}")
    
    # Combination distribution (block_idx, token_type, has_changed)
    combination_counts = Counter(
        (s.get("block_idx", -1), s.get("token_type", "Unknown"), s.get("has_changed", False))
        for s in samples
    )
    print(f"\nUnique (block, token_type, has_changed) combinations: {len(combination_counts)}")
    print(f"Top 10 combinations:")
    for combo, count in combination_counts.most_common(10):
        print(f"  {combo}: {count}")


def validate_dataset(samples: List[Dict]) -> bool:
    """Validate dataset for required fields and consistency."""
    required_fields = [
        "prompt_x", "prompt_xprime", "layer_block", "block_idx",
        "token_position", "token_type", "has_changed", "change_score",
        "original_next_token", "patched_next_token", "aggregate_v"
    ]
    
    errors = []
    for i, sample in enumerate(samples):
        for field in required_fields:
            if field not in sample:
                errors.append(f"Sample {i}: missing field '{field}'")
        
        # Validate aggregate_v is a list
        if "aggregate_v" in sample:
            if not isinstance(sample["aggregate_v"], list):
                errors.append(f"Sample {i}: aggregate_v should be a list")
    
    if errors:
        print(f"Validation errors found: {len(errors)}")
        for error in errors[:10]:  # Print first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    
    print("Dataset validation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate more activation patching data for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name to use for activation patching"
    )
    
    # Data arguments
    parser.add_argument(
        "--counterfact_file",
        type=str,
        default="multi_counterfact.json",
        help="Path to CounterFact dataset JSON file"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=100,
        help="Number of counterfactual pairs to process (0 = all)"
    )
    parser.add_argument(
        "--existing_dataset",
        type=str,
        default="activation_patch_dataset.json",
        help="Path to existing dataset (for resume/merge)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="activation_patch_dataset_extended.json",
        help="Output file path for generated dataset"
    )
    parser.add_argument(
        "--resume_from_existing",
        action="store_true",
        help="Skip pairs that already exist in existing dataset"
    )
    parser.add_argument(
        "--merge_with_existing",
        action="store_true",
        help="Merge new data with existing dataset"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_samples_per_combination",
        type=int,
        default=200,
        help="Maximum samples per (block_idx, token_type, token_pos) combination"
    )
    parser.add_argument(
        "--num_layer_blocks",
        type=int,
        default=4,
        help="Number of layer blocks to divide layers into"
    )
    parser.add_argument(
        "--balance_dataset",
        action="store_true",
        default=True,
        help="Balance dataset across has_changed categories"
    )
    
    # Output format
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["raw", "training"],
        default="training",
        help="Output format: 'raw' (samples) or 'training' (prepared for explainer)"
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*60)
    print("Activation Patching Data Generation")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"CounterFact file: {args.counterfact_file}")
    print(f"Number of pairs: {args.num_pairs if args.num_pairs > 0 else 'all'}")
    print(f"Max samples per combination: {args.max_samples_per_combination}")
    print(f"Output file: {args.output_file}")
    print(f"Output format: {args.output_format}")
    print("="*60)
    
    # Load CounterFact dataset
    print(f"\nLoading CounterFact dataset from {args.counterfact_file}...")
    with open(args.counterfact_file, "r", encoding="utf-8") as f:
        counterfact_dataset = json.load(f)
    
    print(f"Loaded {len(counterfact_dataset)} counterfactual entries")
    
    # Generate x and x' pairs
    print("\nGenerating x and x' pairs...")
    xs, xprimes = generate_all_x_xprime(counterfact_dataset)
    print(f"Generated {len(xs)} pairs")
    
    # Filter existing pairs if resuming
    existing_data = []
    existing_case_ids = set()
    if args.resume_from_existing or args.merge_with_existing:
        existing_data = load_existing_dataset(args.existing_dataset)
        existing_case_ids = get_existing_case_ids(existing_data)
        
        if args.resume_from_existing:
            xs, xprimes = filter_new_pairs(xs, xprimes, existing_case_ids)
    
    # Limit number of pairs if specified
    if args.num_pairs > 0 and len(xs) > args.num_pairs:
        print(f"\nLimiting to {args.num_pairs} pairs (randomly sampled)")
        indices = random.sample(range(len(xs)), args.num_pairs)
        xs = [xs[i] for i in indices]
        xprimes = [xprimes[i] for i in indices]
    
    # Load model
    print(f"\nLoading model: {args.model_name}...")
    model_type = "qwen" if "qwen" in args.model_name.lower() else "llama"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print(f"Model loaded. Type: {model_type}")
    print(f"Number of layers: {len(model.model.layers)}")
    
    # Generate activation patching dataset
    print("\nGenerating activation patching dataset...")
    new_samples = generate_activation_patching_dataset(
        model=model,
        tokenizer=tokenizer,
        xs=xs,
        xprimes=xprimes,
        dataset=counterfact_dataset,
        model_type=model_type,
        max_samples_per_combination=args.max_samples_per_combination
    )
    
    print(f"\nGenerated {len(new_samples)} new samples")
    
    # Balance dataset if requested
    if args.balance_dataset:
        print("\nBalancing dataset...")
        new_samples = balance_dataset(new_samples)
        print(f"After balancing: {len(new_samples)} samples")
    
    # Note: new_samples from generate_activation_patching_dataset are in RAW format
    # They have: prompt_x, prompt_xprime, aggregate_v, etc.
    
    # Detect format of existing data
    existing_is_training = False
    if existing_data and len(existing_data) > 0:
        existing_is_training = "question" in existing_data[0] or "input_text" in existing_data[0]
        if existing_is_training:
            print(f"\nExisting data is in training format ({len(existing_data)} samples)")
        else:
            print(f"\nExisting data is in raw format ({len(existing_data)} samples)")
    
    # Merge with existing if requested
    if args.merge_with_existing and existing_data:
        print(f"\nMerging with existing dataset...")
        
        if args.output_format == "training":
            if existing_is_training:
                # Both should be training format - convert new samples first
                print("Converting new samples to training format...")
                new_samples_training = prepare_explainer_training_data(new_samples)
                all_samples = existing_data + new_samples_training
                print(f"Merged: {len(existing_data)} existing + {len(new_samples_training)} new = {len(all_samples)} total")
            else:
                # Existing is raw, new is raw - convert both to training
                print("Converting both existing and new samples to training format...")
                existing_training = prepare_explainer_training_data(existing_data)
                new_samples_training = prepare_explainer_training_data(new_samples)
                all_samples = existing_training + new_samples_training
                print(f"Merged: {len(existing_training)} existing + {len(new_samples_training)} new = {len(all_samples)} total")
        else:  # output_format == "raw"
            if existing_is_training:
                # Can't convert training back to raw - skip merge
                print("Warning: Cannot convert training format to raw format.")
                print("Skipping merge. Use --output_format training to merge with existing data.")
                all_samples = new_samples
            else:
                # Both are raw - merge directly
                all_samples = existing_data + new_samples
                print(f"Merged: {len(existing_data)} existing + {len(new_samples)} new = {len(all_samples)} total")
    else:
        all_samples = new_samples
    
    # Print statistics (on raw format if available, otherwise on final format)
    print_dataset_statistics(all_samples, "Final Dataset")
    
    # Validate dataset
    print("\nValidating dataset...")
    # Skip validation if data is already in training format (different required fields)
    is_final_training = all_samples and ("question" in all_samples[0] or "input_text" in all_samples[0])
    if is_final_training:
        print("Skipping validation (data is in training format with different field requirements)")
    else:
        if not validate_dataset(all_samples):
            print("Warning: Dataset validation failed, but continuing...")
    
    # Prepare output format
    if args.output_format == "training":
        # Check if data is already in training format
        if is_final_training:
            print("\nData is already in training format - using as-is")
            output_data = all_samples
        else:
            print("\nConverting to training format...")
            output_data = prepare_explainer_training_data(all_samples)
    else:  # output_format == "raw"
        if is_final_training:
            print("\nWarning: Data is in training format but output format is raw.")
            print("Cannot convert training format back to raw. Using training format data.")
        output_data = all_samples
    
    # Save dataset
    print(f"\nSaving dataset to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully saved {len(output_data)} samples to {args.output_file}")
    
    # Print final summary
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Total samples: {len(output_data)}")
    if args.output_format == "training":
        has_changed = sum(1 for d in output_data if d.get("has_changed", False))
        print(f"Has changed: {has_changed} ({100*has_changed/len(output_data):.1f}%)")
    print(f"Output file: {args.output_file}")


if __name__ == "__main__":
    main()

