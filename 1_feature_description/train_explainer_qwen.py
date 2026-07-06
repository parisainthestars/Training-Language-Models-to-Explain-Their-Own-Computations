import os
import json
import math
import random
random.seed(42)

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------------------------------------  
# 0) Paths & basic config                                  
# ------------------------------------------------------- 
import os
CACHE_DIR = "/mnt/raid10/ak-research-01/ak-research-01/codes/.cache"

PROCESSED_ROOT = (
    "/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/"
    "latentqa/1_feature_description/dataset/processed_dataset"
)

# Layers L01–L13, skipping L03
layers = list(range(1, 32))
if 3 in layers:
    layers.remove(3)

TRAIN_JSON_LIST = [
    os.path.join(PROCESSED_ROOT, f"L{layer:02d}", "train.jsonl")
    for layer in layers
]

TEST_JSON_LIST = [
    os.path.join(PROCESSED_ROOT, f"L{layer:02d}", "test.jsonl")
    for layer in layers
]

MODEL_NAME = "Qwen/Qwen3-8B"
MAX_LENGTH = 128   # you can increase if explanations are long 

# -------------------------------------------------------
# 1) Load explainer model and tokenizer
# -------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

model.resize_token_embeddings(len(tokenizer))  # in case we add special tokens
model.train()
dtype = next(model.parameters()).dtype

device = model.device  # "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------
# 2) Special tokens and templates
# -------------------------------------------------------
# Add [s], [e] if needed
SPECIAL_TOKENS = {"additional_special_tokens": ["[s]", "[e]"]}
need_add = any(tok not in tokenizer.get_vocab()
               for tok in SPECIAL_TOKENS["additional_special_tokens"])
if need_add:
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    model.resize_token_embeddings(len(tokenizer))

ID_S = tokenizer.convert_tokens_to_ids("[s]")
ID_E = tokenizer.convert_tokens_to_ids("[e]")

# >>> ADD THIS BLOCK <<<
# Use EOS as padding token (typical for LLaMA-style models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
# <<< END BLOCK >>>

FEATURE_TEMPLATES = [
    "At layer {layer}, [s]v[e] encodes",
    "[s]v[e] activates at layer {layer} for",
    "We can describe [s]v[e] at layer {layer} as encoding",
    "Generate a description of this feature at layer {layer}: [s]v[e].",
    "What does [s]v[e] mean at layer {layer}?",
    "[s]v[e] activates at layer {layer} for inputs with the following features:",
]


def make_feature_prompt(layer: int, template_id: int | None = None) -> str:
    if template_id is None:
        template_id = random.randrange(len(FEATURE_TEMPLATES))
    return FEATURE_TEMPLATES[template_id].format(layer=layer) + " "


def find_v_index(prompt_ids: List[int]) -> int:
    """Find index of the token between [s] and [e] (the 'v' position)."""
    ids = torch.tensor(prompt_ids, dtype=torch.long)

    s_positions = (ids == ID_S).nonzero(as_tuple=True)[0]
    if len(s_positions) == 0:
        raise ValueError("Prompt does not contain [s] token.")
    s_idx = int(s_positions[0].item())

    e_positions = (ids == ID_E).nonzero(as_tuple=True)[0]
    e_positions = e_positions[e_positions > s_idx]
    if len(e_positions) == 0:
        raise ValueError("Prompt does not contain [e] token after [s].")
    e_idx = int(e_positions[0].item())

    mid_positions = list(range(s_idx + 1, e_idx))
    if len(mid_positions) != 1:
        raise ValueError(
            f"Expected exactly one token between [s] and [e], got {len(mid_positions)}."
        )
    return mid_positions[0]



# -------------------------------------------------------
# 3) Dataset
# -------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
        
class FeatureExplanationDataset(Dataset):
    """
    Each item:
        - builds a prompt with [s]v[e] for the given layer
        - concatenates the gold description
        - prepares:
            input_ids   : prompt + description
            attention   : mask
            labels      : -100 on prompt, ids on description
            v           : SAE feature vector (direction)
            v_idx       : index of 'v' token in the sequence
    """
    def __init__(
        self,
        json_paths: List[str],   # <<< list of paths
        tokenizer,
        max_length: int = 128,
    ):
        super().__init__()

        # load and concatenate all jsonl files
        self.data: List[Dict[str, Any]] = []
        for p in json_paths:
            self.data.extend(load_jsonl(p))

        # --- randomly keep 70% ---
        random.shuffle(self.data)                          # in-place shuffle
        k = int(len(self.data) * 0.30)                     # number to keep
        self.data = self.data[:k]                          # subsample

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        layer = int(item["layer"])
        description = item["description"]
        vector = item["vector"]  # list or numpy; we convert to tensor

        # ---- prompt with [s]v[e] ----
        prompt = make_feature_prompt(layer)
        prompt_enc = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_ids = prompt_enc["input_ids"]

        # index of 'v' in the prompt
        v_idx = find_v_index(prompt_ids)

        # ---- explanation tokens ----
        # we add EOS to description so model learns to stop
        expl_text = description + self.tokenizer.eos_token
        expl_enc = self.tokenizer(
            expl_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        expl_ids = expl_enc["input_ids"]

        # ---- full sequence = prompt + explanation ----
        input_ids = prompt_ids + expl_ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
        attention_mask = [1] * len(input_ids)

        # labels: ignore prompt tokens
        labels = [-100] * len(prompt_ids) + expl_ids
        if len(labels) > self.max_length:
            labels = labels[: self.max_length]

        # make sure everything has same length
        assert len(input_ids) == len(labels) == len(attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "v": vector,
            "v_idx": v_idx,
        }


# ---------------------------------------------------------
# 3) Collator that injects v at v_idx
# ---------------------------------------------------------
@dataclass
class ContinuousTokenCollator:
    tokenizer: Any
    model: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn_list      = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels_list    = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        v_list         = [torch.tensor(f["v"], dtype=torch.float32) for f in features]
        v_idx_list     = [int(f["v_idx"]) for f in features]

        # pad input_ids + attention
        batch_enc = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attn_list},
            padding=True,
            return_tensors="pt",
        )
        input_ids = batch_enc["input_ids"]        # [B, L]
        attention_mask = batch_enc["attention_mask"]

        # pad labels
        labels = self.tokenizer.pad(
            {"input_ids": labels_list},
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        # embed and inject v at v_idx
        emb_layer = self.model.get_input_embeddings()
        inputs_embeds = emb_layer(input_ids)  # [B, L, d_model]

        for i, (v, v_idx) in enumerate(zip(v_list, v_idx_list)):
            v = v.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            v = v / (v.norm() + 1e-8)
            if v_idx < inputs_embeds.size(1):
                inputs_embeds[i, v_idx, :] = v

        return {
            "inputs_embeds": inputs_embeds.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }


train_dataset = FeatureExplanationDataset(TRAIN_JSON_LIST, tokenizer, MAX_LENGTH)
eval_dataset = FeatureExplanationDataset(TEST_JSON_LIST, tokenizer, MAX_LENGTH)

data_collator = ContinuousTokenCollator(tokenizer=tokenizer, model=model)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=data_collator,
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=data_collator,
)


# ---------------------------------------------------------
# 4) Simple training loop (cross-entropy on explanation)
# ---------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (step + 1) % 50 == 0:
            avg = total_loss / 50
            print(f"Epoch {epoch+1} | Step {step+1} | Loss {avg:.4f}")
            total_loss = 0.0

    # quick eval after each epoch
    model.eval()
    eval_loss = 0.0
    eval_steps = 0
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            eval_loss += outputs.loss.item()
            eval_steps += 1
    if eval_steps > 0:
        print(f"[Eval] Epoch {epoch+1} | Loss {eval_loss / eval_steps:.4f}")


DATA_DIR = "/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/latentqa/1_feature_description/model"

# Save fine-tuned explainer
model_name = "qwen_explainer_L01_L31_ckpt"
os.makedirs(os.path.join(DATA_DIR, model_name), exist_ok=True)
model.save_pretrained(os.path.join(DATA_DIR, model_name))
tokenizer.save_pretrained(os.path.join(DATA_DIR, model_name))