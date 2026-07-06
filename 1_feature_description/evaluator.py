# from feature_description_dataset import *
from torch.utils.data import Dataset
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple
import torch
from contextlib import contextmanager
import random
import json
from tqdm import tqdm
from LM_judge import *



import os
CACHE_DIR = "/mnt/raid10/ak-research-01/ak-research-01/codes/.cache"

PROCESSED_ROOT = (
    "/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/"
    "latentqa/1_feature_description/dataset/processed_dataset"
)

MODEL_NAME = "explainer_L01_L31_ckpt"   #    
MAX_LENGTH = 128                        



DATA_DIR = "/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/latentqa/1_feature_description/model"
CKPT_DIR = os.path.join(DATA_DIR, MODEL_NAME)

os.makedirs(f"{CKPT_DIR}/results", exist_ok=True)

# 1) Load tokenizer and model from the checkpoint directory
tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    CKPT_DIR,
    torch_dtype=torch.bfloat16,     # match what you used in training
    device_map="auto",              # or device="cuda"
)

# 2) Ensure [s] and [e] are special tokens
special_tokens = {"additional_special_tokens": ["[s]", "[e]"]}

need_add = any(tok not in tokenizer.get_vocab()
               for tok in special_tokens["additional_special_tokens"])
if need_add:
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

ID_S = tokenizer.convert_tokens_to_ids("[s]")
ID_E = tokenizer.convert_tokens_to_ids("[e]")

# 3) Paper templates with [s]v[e]
FEATURE_TEMPLATES = [
    "At layer {layer}, [s]v[e] encodes",
    "[s]v[e] activates at layer {layer} for",
    "We can describe [s]v[e] at layer {layer} as encoding",
    "Generate a description of this feature at layer {layer}: [s]v[e].",
    "What does [s]v[e] mean at layer {layer}?",
    "[s]v[e] activates at layer {layer} for inputs with the following features:",
]


# 2) Ensure pad token is set (needed for padding and generation)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

model.eval()   # for inference
device = model.device
dtype = next(model.parameters()).dtype

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


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_feature_prompt(layer: int, template_id: int | None = None) -> str:
    if template_id is None:
        template_id = random.randrange(len(FEATURE_TEMPLATES))
    return FEATURE_TEMPLATES[template_id].format(layer=layer) + " "

def normalize_feature(v: torch.Tensor) -> torch.Tensor:
    v = v.to(device=device, dtype=dtype)
    return v / (v.norm() + 1e-8)


@contextmanager
def patch_at_v_between_s_e(model, input_ids: torch.Tensor, v: torch.Tensor):
    """
    Find a span [s] v [e] in the tokenized *prompt* and replace the embedding of
    the token 'v' (the one between [s] and [e]) with the continuous vector v.
    This patch is applied only on the first forward pass (full prompt),
    and skipped on later generation steps (seq_len = 1).
    """
    emb = model.get_input_embeddings()
    v = v.to(device=emb.weight.device, dtype=emb.weight.dtype)

    ids = input_ids[0]  # [seq_len] for the *prompt*

    # locate [s]
    s_positions = (ids == ID_S).nonzero(as_tuple=True)[0]
    if len(s_positions) == 0:
        raise ValueError("Prompt does not contain [s] token.")
    s_idx = int(s_positions[0].item())

    # locate [e] AFTER [s]
    e_positions = (ids == ID_E).nonzero(as_tuple=True)[0]
    e_positions = e_positions[e_positions > s_idx]
    if len(e_positions) == 0:
        raise ValueError("Prompt does not contain [e] token after [s].")
    e_idx = int(e_positions[0].item())

    # tokens strictly between [s] and [e] → expect just 'v'
    mid_positions = list(range(s_idx + 1, e_idx))
    if len(mid_positions) != 1:
        raise ValueError(
            f"Expected exactly one token between [s] and [e], got {len(mid_positions)}."
        )
    v_idx = mid_positions[0]

    patched_once = {"done": False}  # mutable flag closed over by hook

    def hook(module, inputs, output):
        out = output.clone()  # [batch, seq_len, d_model]
        seq_len = out.size(1)

        # Only patch on first call with full prompt where v_idx is in range
        if (not patched_once["done"]) and (seq_len > v_idx):
            out[:, v_idx, :] = v
            patched_once["done"] = True

        return out

    handle = emb.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def explain_feature_with_llama(v: torch.Tensor,
                               layer: int,
                               template_id: int | None = None,
                               max_new_tokens: int = 96) -> str:
    """
    v:      SAE feature direction in residual space, shape [hidden_size] (4096).
    layer:  layer index ℓ for the textual prompt.
    """
    prompt = make_feature_prompt(layer, template_id)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    v_norm = normalize_feature(v)

    with patch_at_v_between_s_e(model, input_ids, v_norm):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    gen_ids = out[0, prompt_len:]          # continuation only
    explanation_new = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return tokenizer.decode(out[0], skip_special_tokens=True), explanation_new

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


# Layers L01–L13, skipping L03
layers = list(range(1, 32))
if 3 in layers:
    layers.remove(3)


out_path = f"{CKPT_DIR}/results/results.jsonl"

results = []

for layer in tqdm(layers):

    TEST_JSON_LIST = os.path.join(PROCESSED_ROOT, f"L{layer:02d}", "test.jsonl")
    eval_dataset = FeatureExplanationDataset([TEST_JSON_LIST,], tokenizer, MAX_LENGTH)

    for data in eval_dataset.data:
        v = torch.tensor(data["vector"], dtype=torch.float32)
        layer_id = int(data["layer"])

        explanation, explanation_new = explain_feature_with_llama(v, layer_id)

        result_item = {
            "predicted_explanation": explanation_new,
            "true_explanation": data["description"],
            "layer": layer_id,
            "f_index": int(data["index"]),
            "score": lm_judge_score(explanation_new, data["description"]),  
        }
        results.append(result_item)

# write per-layer JSONL file
with open(out_path, "w", encoding="utf-8") as f_out:
    for r in results:
        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
