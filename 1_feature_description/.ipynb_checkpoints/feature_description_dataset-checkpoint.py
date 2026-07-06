# feature_description_dataset.py

import os
import json
import random
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any

from torch.utils.data import Dataset

import torch
from sae_lens import SAE
from neuronpedia.np_sae_feature import SAEFeature

import time
import requests

os.environ["NEURONPEDIA_API_KEY"] = "sk-np-pa76I7f2OTMBhSi0Q4oZHwY04mjEy2mX6ARdXDB2EBc0"

# -------------------------
# Config
# -------------------------

SAE_ROOT_DIR = (
    "/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/"
    "latentqa/1_feature_description/dataset/llamascope_LXR_32x"
)

PROCESSED_ROOT_DIR = (
    "/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/"
    "latentqa/1_feature_description/dataset/processed_dataset"
)

MODEL_ID = "llama3.1-8b"          # Neuronpedia model id
SOURCE_TEMPLATE = "{layer}-llamascope-res-131k"  # Neuronpedia source per layer


# -------------------------
# Helpers
# -------------------------

def _ensure_neuronpedia_api_key(env_var: str = "NEURONPEDIA_API_KEY") -> None:
    if env_var not in os.environ or not os.environ[env_var]:
        raise RuntimeError(
            f"{env_var} is not set. Please export your Neuronpedia API key, e.g.\n"
            f"  export {env_var}=<your_key_here>"
        )


def safe_get_feature(model_id, source, feature_index, retries=3, delay=5):
    last_err = None
    for i in range(retries):
        try:
            return SAEFeature.get(
                model_id=model_id,
                source=source,
                index=str(feature_index),
            )
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            print(f"[WARN] ReadTimeout on attempt {i+1}/{retries}, retrying in {delay}s...")
            time.sleep(delay)
    raise last_err


def load_sae_for_layer(
    layer: int,
    sae_root_dir: str = SAE_ROOT_DIR,
    template: str = "Llama3_1-8B-Base-L{layer}R-32x",
    device: Optional[str] = None,
) -> SAE:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_file = template.format(layer=layer)
    sae_path = os.path.join(sae_root_dir, layer_file)
    sae = SAE.load_from_disk(sae_path, device=device)
    return sae


def fetch_neuronpedia_explanation(
    layer: int,
    feature_index: int,
    model_id: str = MODEL_ID,
    source_template: str = SOURCE_TEMPLATE,
) -> Optional[str]:
    _ensure_neuronpedia_api_key()
    source = source_template.format(layer=layer)

    # feat = SAEFeature.get(
    #     model_id=model_id,
    #     source=source,
    #     index=str(feature_index),
    # )

    feat = safe_get_feature(model_id, source, feature_index)

    data = feat.jsonData if isinstance(feat.jsonData, dict) else json.loads(feat.jsonData)
    explanations = data.get("explanations", [])
    if not explanations:
        return None
    desc = explanations[0].get("description", "").strip()
    return desc or None


# -------------------------
# Per-layer dataset builder
# -------------------------

def build_layer_dataset(
    layer: int,
    n_test: int = 50,
    n_train_per_layer: int = 2000,
    seed: int = 0,
    device: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """
    Build train/test examples for a single layer.

    Returns:
        train_examples, test_examples, test_ids
    """
    sae = load_sae_for_layer(layer=layer, device=device)
    d_sae = sae.cfg.d_sae

    all_indices = list(range(d_sae))
    rng = random.Random(seed + layer)
    rng.shuffle(all_indices)

    # --- Select test indices with valid explanations ---
    test_ids: List[int] = []
    test_examples: List[Dict[str, Any]] = []

    for idx in all_indices:
        if len(test_ids) >= n_test:
            break
        desc = fetch_neuronpedia_explanation(layer=layer, feature_index=idx)
        if desc is None:
            continue

        v = sae.W_dec.T[:, idx].detach().cpu().tolist()

        example = {
            "layer": layer,
            "index": idx,
            "description": desc,
            "vector": v,
        }
        test_examples.append(example)
        test_ids.append(idx)

    # Remaining indices for training
    remaining_indices = [i for i in all_indices if i not in test_ids]

    train_examples: List[Dict[str, Any]] = []
    for idx in remaining_indices:
        if len(train_examples) >= n_train_per_layer:
            break
        desc = fetch_neuronpedia_explanation(layer=layer, feature_index=idx)
        if desc is None:
            continue

        v = sae.W_dec.T[:, idx].detach().cpu().tolist()

        example = {
            "layer": layer,
            "index": idx,
            "description": desc,
            "vector": v,
        }
        train_examples.append(example)

    return train_examples, test_examples, test_ids


# -------------------------
# All-layers builder & saver
# -------------------------

def build_and_save_all_layers_datasets(
    layers: Optional[List[int]] = None,
    n_test: int = 50,
    n_train_per_layer: int = 2000,
    base_seed: int = 0,
    device: Optional[str] = None,
) -> None:
    """
    Build datasets for layers L01..L31 (skipping L03) and save to disk.

    For each layer ℓ:
      - directory:  PROCESSED_ROOT_DIR / f"L{ℓ:02d}"
      - files:
          - LXX_test_ids.txt  (one feature index per line)
          - train.jsonl       (one JSON per line)
          - test.jsonl
    """
    if layers is None:
        layers = list(range(2, 32))  # 1..31 inclusive
        if 3 in layers:
            layers.remove(3)         # skip layer 3 as in the paper

    os.makedirs(PROCESSED_ROOT_DIR, exist_ok=True)

    for layer in layers:
        print(f"=== Building dataset for layer {layer} ===")
        layer_dirname = f"L{layer:02d}"
        out_dir = os.path.join(PROCESSED_ROOT_DIR, layer_dirname)
        os.makedirs(out_dir, exist_ok=True)

        train_examples, test_examples, test_ids = build_layer_dataset(
            layer=layer,
            n_test=n_test,
            n_train_per_layer=n_train_per_layer,
            seed=base_seed,
            device=device,
        )

        # Save test IDs
        test_ids_path = os.path.join(out_dir, f"{layer_dirname}_test_ids.txt")
        with open(test_ids_path, "w", encoding="utf-8") as f_ids:
            for idx in test_ids:
                f_ids.write(f"{idx}\n")

        # Save train.jsonl
        train_path = os.path.join(out_dir, "train.jsonl")
        with open(train_path, "w", encoding="utf-8") as f_train:
            for ex in train_examples:
                f_train.write(json.dumps(ex, ensure_ascii=False) + "\n")

        # Save test.jsonl
        test_path = os.path.join(out_dir, "test.jsonl")
        with open(test_path, "w", encoding="utf-8") as f_test:
            for ex in test_examples:
                f_test.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(
            f"Layer {layer}: saved {len(train_examples)} train / "
            f"{len(test_examples)} test examples to {out_dir}"
        )


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

# -------------------------
# CLI entry point
# -------------------------

if __name__ == "__main__":
    # Example: build datasets for all layers with 50 test and 2000 train per layer
    build_and_save_all_layers_datasets(
        n_test=50,
        n_train_per_layer=1000,
        base_seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
