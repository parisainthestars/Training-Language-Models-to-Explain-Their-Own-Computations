from sae_lens import SAE
import torch, os

local_dir = "/mnt/raid10/ak-research-01/ak-research-01/codes/steer-vector/latentqa/1_feature_description/dataset/llamascope_LXR_32x"

# 2) Inspect files manually first and set this accordingly:
layer_file = "Llama3_1-8B-Base-L16R-32x" # "Llama3_1-8B-Base-L2R-32x/checkpoints/final.safetensors"  # <-- change to actual filename
sae_path = os.path.join(local_dir, layer_file)

device = "cuda" if torch.cuda.is_available() else "cpu"
sae = SAE.load_from_disk(sae_path, device=device)


# sae = SAE.load_from_disk(sae_path, device=device)
print("Loaded SAE with d_in:", sae.cfg.d_in, "d_sae:", sae.cfg.d_sae)


print(sae)

print("Loaded SAE:", sae.cfg)
print("Decoder shape:", sae.W_dec.shape)

# 3) Grab a single feature vector
feature_index = 23784 # and layer 16
v = sae.W_dec.T[:, feature_index]

print("Feature vector shape:", v.shape)
print("L2 norm:", torch.norm(v).item())
