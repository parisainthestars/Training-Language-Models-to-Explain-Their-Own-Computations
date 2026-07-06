#!/usr/bin/env bash

BASE_DIR="/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/latentqa/1_feature_description/dataset/llamascope_LXR_32x"

for L in $(seq 0 31); do
    # skip layer 16
    if [ "$L" -eq 16 ]; then
        echo "Skipping layer $L"
        continue
    fi

    LAYER_DIR="${BASE_DIR}/Llama3_1-8B-Base-L${L}R-32x"
    echo "Processing layer $L: $LAYER_DIR"

    if [ ! -d "$LAYER_DIR" ]; then
        echo "  Directory not found, skipping."
        continue
    fi

    # 1) Rename hyperparams.json -> cfg.json
    if [ -f "${LAYER_DIR}/hyperparams.json" ]; then
        mv "${LAYER_DIR}/hyperparams.json" "${LAYER_DIR}/cfg.json"
        echo "  Renamed hyperparams.json -> cfg.json"
    else
        echo "  hyperparams.json not found (maybe already renamed)."
    fi

    # 2) Move checkpoints/final.safetensors up and rename
    if [ -f "${LAYER_DIR}/checkpoints/final.safetensors" ]; then
        mv "${LAYER_DIR}/checkpoints/final.safetensors" "${LAYER_DIR}/final.safetensors"
        echo "  Moved checkpoints/final.safetensors -> final.safetensors"

        # Also create sae_weights.safetensors
        cp "${LAYER_DIR}/final.safetensors" "${LAYER_DIR}/sae_weights.safetensors"
        echo "  Copied final.safetensors -> sae_weights.safetensors"
    else
        echo "  checkpoints/final.safetensors not found (maybe already moved)."
    fi

    # 3) Edit cfg.json: change d_model -> d_in
    if [ -f "${LAYER_DIR}/cfg.json" ]; then
        CFG_PATH="${LAYER_DIR}/cfg.json" python - << 'EOF'
import json, os

path = os.environ["CFG_PATH"]
with open(path, "r") as f:
    cfg = json.load(f)

if "d_model" in cfg:
    cfg["d_in"] = cfg.pop("d_model")

with open(path, "w") as f:
    json.dump(cfg, f, indent=4)
EOF
        echo "  Updated cfg.json (d_model -> d_in)."
    else
        echo "  cfg.json not found, skipping JSON edit."
    fi

    echo
done
