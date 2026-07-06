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
CACHE_DIR = "/mnt/raid10/ak-research-01/ak-research-01/codes/.cache"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from contextlib import contextmanager
import random


# 3) Paper templates with [s]v[e]
FEATURE_TEMPLATES = [
    "At layer {layer}, [s]v[e] encodes",
    "[s]v[e] activates at layer {layer} for",
    "We can describe [s]v[e] at layer {layer} as encoding",
    "Generate a description of this feature at layer {layer}: [s]v[e].",
    "What does [s]v[e] mean at layer {layer}?",
    "[s]v[e] activates at layer {layer} for inputs with the following features:",
]



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
        layers = list(range(27, 32))  # 1..31 inclusive
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
