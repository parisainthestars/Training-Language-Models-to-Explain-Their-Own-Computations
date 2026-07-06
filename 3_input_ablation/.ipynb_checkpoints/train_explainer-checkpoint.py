#!/usr/bin/env python
"""
Fine-tune meta-llama/Meta-Llama-3.1-8B-Instruct as the explainer
for the Input Ablation task.

It expects a JSONL dataset where each line is a dict with keys:
    - "explainer_prompt": str
    - "explainer_output": str

The loss is applied **only** on the assistant part
(i.e., explainer_output); the prompt tokens are masked with -100.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)



# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

DATA_PATH = "/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/latentqa/3_input_ablation/generated_dataset/input_ablation_llama_instruct_train.jsonl"
CACHE_DIR = "/mnt/raid10/ak-research-01/ak-research-01/codes/.cache"
OUTPUT_DIR = "/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/latentqa/3_input_ablation/models"

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MAX_LENGTH = 512          # truncate long examples
VAL_SPLIT = 0.05          # 5% of data for validation

BATCH_SIZE = 16            # per-device batch size (8B model is heavy)
GRAD_ACCUM = 8            # effective batch size = BATCH_SIZE * GRAD_ACCUM
NUM_EPOCHS = 3
LR = 5e-6
WEIGHT_DECAY = 0.01

SEED = 42


# ------------------------------------------------------------------
# LOAD TOKENIZER & MODEL
# ------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Make sure we have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    cache_dir=CACHE_DIR
)
model.resize_token_embeddings(len(tokenizer))



# ------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------

# Single JSONL file → treat it as one split called "train"
raw_ds = load_dataset(
    "json",
    data_files={"train": DATA_PATH},
)["train"]

ds = raw_ds.train_test_split(test_size=VAL_SPLIT, seed=SEED)
train_ds = ds["train"]
val_ds = ds["test"]


def preprocess_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build input_ids, attention_mask, and labels.

    - full_text = explainer_prompt + explainer_output + eos
    - labels = -100 for prompt tokens; real ids for output tokens.
    """
    prompt = example["explainer_prompt"]
    target = example["explainer_output"]

    # Text for model
    full_text = prompt + target + tokenizer.eos_token

    # Tokenize both prompt and full text without extra special tokens
    prompt_enc = tokenizer(
        prompt,
        add_special_tokens=False,
    )
    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=MAX_LENGTH,
        truncation=True,
    )

    input_ids = full_enc["input_ids"]
    attn_mask = full_enc["attention_mask"]

    prompt_len = len(prompt_enc["input_ids"])
    # If truncation cut off part of the prompt, clamp
    prompt_len = min(prompt_len, len(input_ids))

    # Build labels: -100 for prompt part, real ids for answer part
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    assert len(labels) == len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
    }


print("Tokenizing dataset...")
train_tokenized = train_ds.map(
    preprocess_example,
    remove_columns=train_ds.column_names,
    desc="Tokenizing train",
)
val_tokenized = val_ds.map(
    preprocess_example,
    remove_columns=val_ds.column_names,
    desc="Tokenizing val",
)



# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=0.03,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),   # use bfloat16 if possible
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    report_to="wandb",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=default_data_collator,
)

trainer.train()

# Save final model + tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training complete. Model saved to {OUTPUT_DIR}")