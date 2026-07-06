import json
import argparse
from activation_patching import (
    generate_all_x_xprime,
    generate_activation_patching_dataset,
    balance_dataset,
    prepare_explainer_training_data,
    analyze_dataset
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

def main():
    parser = argparse.ArgumentParser(
        description="Generate activation patching dataset for explainer training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data files
    parser.add_argument("--input_file", type=str, default="multi_counterfact.json",
                       help="Input counterfactual dataset file")
    parser.add_argument("--output_file", type=str, default="activation_patch_dataset.json",
                       help="Output dataset file")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Target model to perform activation patching on")
    
    # Generation parameters
    parser.add_argument("--max_samples_per_combination", type=int, default=20,
                       help="Maximum samples per (block, token_type, position) combination")
    parser.add_argument("--max_pairs", type=int, default=None,
                       help="Maximum number of counterfactual pairs to process (None = all)")
    
    # Analysis only
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze existing dataset, don't generate new data")
    
    # Resume/append
    parser.add_argument("--append", action="store_true",
                       help="Append to existing dataset instead of overwriting")
    
    args = parser.parse_args()
    
    # Analyze existing dataset if requested
    if args.analyze_only:
        print("=" * 60)
        print("ANALYZING EXISTING DATASET")
        print("=" * 60)
        analyze_dataset(args.output_file)
        return
    
    # Load counterfactual dataset
    print("=" * 60)
    print("LOADING COUNTERFACTUAL DATASET")
    print("=" * 60)
    print(f"Loading from: {args.input_file}")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} counterfactual entries")
    except FileNotFoundError:
        print(f"Error: Input file {args.input_file} not found!")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Generate x and x' pairs
    xs, xprimes = generate_all_x_xprime(dataset)
    print(f"Generated {len(xs)} counterfactual pairs")
    
    # Limit pairs if specified
    if args.max_pairs is not None:
        xs = xs[:args.max_pairs]
        xprimes = xprimes[:args.max_pairs]
        print(f"Limited to {len(xs)} pairs for processing")
    
    # Load model
    print("\n" + "=" * 60)
    print("LOADING TARGET MODEL")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    model_type = "qwen" if "qwen" in args.model_name.lower() else "llama"
    print(f"Model type: {model_type}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate activation patching dataset
    print("\n" + "=" * 60)
    print("GENERATING ACTIVATION PATCHING DATASET")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Max samples per combination: {args.max_samples_per_combination}")
    print(f"  Number of pairs: {len(xs)}")
    print(f"  Model: {args.model_name}")
    
    samples = generate_activation_patching_dataset(
        model, tokenizer, xs, xprimes, dataset,
        model_type=model_type,
        max_samples_per_combination=args.max_samples_per_combination
    )
    
    print(f"\nGenerated {len(samples)} raw samples")
    
    # Balance dataset
    print("\nBalancing dataset...")
    balanced_samples = balance_dataset(samples)
    print(f"Balanced dataset: {len(balanced_samples)} samples")
    
    # Prepare training data
    print("\nPreparing training data for explainer...")
    training_data = prepare_explainer_training_data(balanced_samples)
    
    # Handle append mode
    if args.append:
        print(f"\nAppending to existing dataset: {args.output_file}")
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"Found {len(existing_data)} existing samples")
            training_data = existing_data + training_data
            print(f"Total samples after append: {len(training_data)}")
        except FileNotFoundError:
            print("No existing dataset found, creating new one")
        except Exception as e:
            print(f"Warning: Could not load existing dataset: {e}")
            print("Creating new dataset instead")
    
    # Save dataset
    print(f"\nSaving dataset to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(training_data)} training samples")
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    analyze_dataset(args.output_file)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Dataset saved to: {args.output_file}")
    print(f"Total samples: {len(training_data)}")
    print("\nNext steps:")
    print("  1. Review the dataset statistics above")
    print("  2. Train explainer model:")
    print(f"     python activation_patching.py train --dataset_file {args.output_file}")

if __name__ == "__main__":
    import torch
    main()


