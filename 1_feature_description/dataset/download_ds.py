from huggingface_hub import snapshot_download

# Llama-3.1-8B, residual stream SAEs, 32x expansion (32K features)
local_dir = snapshot_download(
    repo_id="fnlp/Llama3_1-8B-Base-LXR-32x",
    local_dir="llamascope_LXR_32x",
    local_dir_use_symlinks=False,
)
print(local_dir)

