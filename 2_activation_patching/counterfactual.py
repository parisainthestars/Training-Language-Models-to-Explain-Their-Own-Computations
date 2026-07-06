# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # -----------------------------
# # Load first 2 samples
# # -----------------------------
# def load_samples(x_path="x_samples.json", xprime_path="xprime_samples.json", n=2):
#     with open(x_path, "r", encoding="utf-8") as f:
#         xs = json.load(f)[:n]
#     with open(xprime_path, "r", encoding="utf-8") as f:
#         xprimes = json.load(f)[:n]
#     return xs, xprimes

# # -----------------------------
# # Forward pass to get logits and hidden
# # -----------------------------
# def get_hidden_and_output(model, tokenizer, prompt, token_position, layer):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#     logits = outputs.logits[0].detach().cpu().tolist()
#     hidden_states = outputs.hidden_states
#     h_l_t = hidden_states[layer][0, token_position, :].detach().clone()
#     return logits, h_l_t

# # -----------------------------
# # Patch hidden and forward
# # -----------------------------
# def patch_hidden_and_forward(model, tokenizer, prompt_x, h_t_xprime, token_position, layer):
#     inputs = tokenizer(prompt_x, return_tensors="pt").to(model.device)

#     def hook(module, input, output):
#         output[:, token_position, :] = h_t_xprime.to(output.device)
#         return output

#     handle = model.model.layers[layer].register_forward_hook(hook)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     handle.remove()

#     return outputs.logits[0].detach().cpu().tolist()

# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     model_name = "Qwen/Qwen3-8B"
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True
#     )
#     model.eval()

#     xs, xprimes = load_samples()

#     dataset_patch = []

#     # Example: test token_position=3, layer=5
#     token_position = 3
#     layer = 5

#     for x, xprime in zip(xs, xprimes):
#         M_x, _ = get_hidden_and_output(model, tokenizer, x["prompt"], token_position, layer)
#         _, h_t_xprime = get_hidden_and_output(model, tokenizer, xprime["prompt"], token_position, layer)
#         M_x_h_t_xprime = patch_hidden_and_forward(model, tokenizer, x["prompt"], h_t_xprime, token_position, layer)

#         dataset_patch.append({
#             "case_id": x["case_id"],
#             "prompt": x["prompt"],
#             "subject": x["subject"],
#             "expected_factual": x["expected_object"],
#             "expected_counterfactual": xprime["expected_object"],
#             "M_x": M_x,
#             "M_x_h_t_xprime": M_x_h_t_xprime,
#             "token_position": token_position,
#             "layer": layer
#         })

#     # Save small test dataset
#     with open("activation_patch_test.json", "w", encoding="utf-8") as f:
#         json.dump(dataset_patch, f, indent=2, ensure_ascii=False)

#     print("Saved activation patching test for 2 samples to activation_patch_test.json")

# if __name__ == "__main__":
#     main()
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Multi-choice options (distractors only)
# -----------------------------
def get_distractor_options(entry, dataset, num_options=6):
    """
    Generate distractors for a given entry:
    - Only distractors (exclude factual and counterfactual)
    """
    true_ans = entry["expected_factual"]
    counterfactual_ans = entry["expected_counterfactual"]

    # Collect distractors with the same relation
    relation_id = entry["relation_id"]
    distractors = [
        e["expected_factual"]
        for e in dataset
        if e["relation_id"] == relation_id
        and e["expected_factual"] not in [true_ans, counterfactual_ans]
    ]
    random.shuffle(distractors)
    return distractors[:num_options]

def build_prompt_with_x_xprime(x_entry, xprime_entry, dataset, num_options=6):
    # Get only distractors for answer options
    options = get_distractor_options(x_entry, dataset, num_options)

    prompt_lines = [
        x_entry["prompt"].format(x_entry["subject"]),     # factual sentence
        xprime_entry["prompt"].format(xprime_entry["subject"]),  # counterfactual sentence
        "Respond with one of " + " or ".join(options) + " and nothing else."
    ]
    return "\n".join(prompt_lines)

# -----------------------------
# Generate x and x' entries
# -----------------------------
def generate_x_xprime(entry):
    rr = entry["requested_rewrite"]
    subject = rr["subject"]
    prompt_template = rr["prompt"]

    x = {
        "prompt": prompt_template,
        "subject": subject,
        "expected_factual": rr["target_true"]["str"],
        "expected_counterfactual": rr["target_new"]["str"],
        "relation_id": rr["relation_id"],
        "case_id": entry["case_id"],
        "type": "factual"
    }

    x_prime = {
        "prompt": prompt_template,
        "subject": subject,
        "expected_factual": rr["target_true"]["str"],
        "expected_counterfactual": rr["target_new"]["str"],
        "relation_id": rr["relation_id"],
        "case_id": entry["case_id"],
        "type": "counterfactual"
    }

    return x, x_prime

def generate_all_x_xprime(dataset):
    xs, xprimes = [], []
    for entry in dataset:
        x, xp = generate_x_xprime(entry)
        xs.append(x)
        xprimes.append(xp)
    return xs, xprimes

# -----------------------------
# Forward pass & activation patching
# -----------------------------
def get_hidden_and_output(model, tokenizer, prompt, token_position, layer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits[0].detach().cpu().tolist()
    hidden_states = outputs.hidden_states
    h_l_t = hidden_states[layer][0, token_position, :].detach().clone()
    return logits, h_l_t

def patch_hidden_and_forward(model, tokenizer, prompt_x, h_t_xprime, token_position, layer):
    inputs = tokenizer(prompt_x, return_tensors="pt").to(model.device)

    def hook(module, input, output):
        output[:, token_position, :] = h_t_xprime.to(output.device)
        return output

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        outputs = model(**inputs)
    handle.remove()

    return outputs.logits[0].detach().cpu().tolist()

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # Load dataset
    with open("multi_counterfact.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Generate x and x' entries
    xs, xprimes = generate_all_x_xprime(dataset)

    # Load model
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    dataset_patch = []

    # Example: only first 2 samples for testing
    for x, xprime in zip(xs[:2], xprimes[:2]):
        token_position = 3
        layer = 5

        # Build multi-line prompt including both x and x'
        options = get_distractor_options(x, xs, num_options=6)
        prompt_mc = build_prompt_with_x_xprime(x, xprime, xs, num_options=6)  # Pass xs as dataset

        # Forward pass
        M_x, _ = get_hidden_and_output(model, tokenizer, prompt_mc, token_position, layer)
        _, h_t_xprime = get_hidden_and_output(model, tokenizer, prompt_mc, token_position, layer)
        M_x_h_t_xprime = patch_hidden_and_forward(model, tokenizer, prompt_mc, h_t_xprime, token_position, layer)

        dataset_patch.append({
            "case_id": x["case_id"],
            "prompt": prompt_mc,
            "subject": x["subject"],
            "expected_factual": x["expected_factual"],
            "expected_counterfactual": x["expected_counterfactual"],
            "M_x": M_x,
            "M_x_h_t_xprime": M_x_h_t_xprime,
            "token_position": token_position,
            "layer": layer
        })

    # Save results
    with open("activation_patch_mc_prompt_test.json", "w", encoding="utf-8") as f:
        json.dump(dataset_patch, f, indent=2, ensure_ascii=False)

    print("Saved activation patching multi-line prompt test for 2 samples.")

if __name__ == "__main__":
    main()
