#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoE Expert Trace → JSON (Single Prompt, Qwen / Mixtral-MoE, vLLM-V1)
---------------------------------------------------------------------
Example:
    python moe_trace_single.py \
        --model /path/to/moe-model \
        --prompt "Hello, please write a short introduction about Mixture-of-Experts."

Output:
    A single JSON line printed to stdout, containing:
        - input_tokens
        - output_text
        - output_tokens
        - input_expert_usage / input_avg_probs
        - output_expert_usage / output_avg_probs
"""

import os
import re
import json
import torch
import random
import argparse
import pathlib
import numpy as np
from vllm import LLM, SamplingParams

# Environment configuration
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

# CLI arguments
parser = argparse.ArgumentParser(description="Trace MoE expert routing for a single prompt.")
parser.add_argument("--model", required=True, help="Path to the model directory.")
parser.add_argument("--prompt", required=True, help="Input text prompt.")
parser.add_argument("--max_new", type=int, default=128, help="Maximum number of generated tokens.")
parser.add_argument("--dtype", default="bfloat16", help="vLLM dtype (e.g., float16 or bfloat16).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
args = parser.parse_args()

# Deterministic seed setup
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

load_cfg = lambda d: json.load(open(pathlib.Path(d) / "config.json", encoding="utf-8"))
arch_of = lambda c: (
    "mixtral"
    if "mixtral" in c.get("model_type", "") or "mistral" in c.get("model_type", "")
    else "qwen"
)
gate_pat = (
    lambda a: re.compile(r"\.block_sparse_moe\.(?:gate|router)$")
    if a == "mixtral"
    else re.compile(r"\.mlp\.gate$")
)
get_int = lambda c, *ks: int(next((c[k] for k in ks if k in c), 1))
def get_num_experts(cfg: dict) -> int:
    """Extract the total number of experts from config.json."""
    if "num_experts" in cfg:
        return int(cfg["num_experts"])
    if "num_local_experts" in cfg:
        return int(cfg["num_local_experts"])
    if "n_routed_experts" in cfg:  # e.g., DeepSeek / MoE-AI style configs
        routed = int(cfg["n_routed_experts"])
        shared = int(cfg.get("n_shared_experts", 0))
        return routed + shared  # e.g., 64 + 2 = 66
    raise KeyError("Could not find expert count field in model config.")

# Load model configuration
cfg = load_cfg(args.model)
arch = arch_of(cfg)
PAT = gate_pat(arch)
L = cfg["num_hidden_layers"]
E = get_num_experts(cfg)
TK = get_int(cfg, "num_experts_per_tok", "router_top_k")

# Initialize vLLM
llm_kwargs = {
    "model": args.model,
    "enforce_eager": True,
    "trust_remote_code": True,
    "dtype": args.dtype,
    "seed": args.seed,
    "gpu_memory_utilization": 0.95,
    "max_model_len": 65536,  # limit context to 64K for safety
}

print(f"[Info] Model: {args.model}")
print(f"[Info] GPU memory utilization: {llm_kwargs['gpu_memory_utilization']}")
llm = LLM(**llm_kwargs)
params = SamplingParams(max_tokens=args.max_new, temperature=0.0)

# Install forward hooks to record expert routing
def _install(exe):
    """Attach forward hooks to gate modules to trace expert usage."""
    exe._trace_in = torch.zeros(L, E, dtype=torch.int32, device=exe.device)
    exe._trace_out = torch.zeros(L, E, dtype=torch.int32, device=exe.device)
    exe._sum_in = torch.zeros(L, E, dtype=torch.float32, device=exe.device)
    exe._sum_out = torch.zeros(L, E, dtype=torch.float32, device=exe.device)
    exe._tok_seen = 0
    exe._plen = 0

    def mk(li):
        last_layer = li == L - 1

        def hook(_m, _in, out):
            logits = out[0] if isinstance(out, (tuple, list)) else out
            prob = torch.softmax(logits, -1)
            v, idx = prob.topk(TK, -1)

            flat_idx = idx.view(-1, TK)
            flat_val = v.view(-1, TK)
            for t in range(flat_idx.size(0)):
                g = exe._tok_seen + t
                target = (exe._trace_in, exe._sum_in) if g < exe._plen else (exe._trace_out, exe._sum_out)
                for k in range(TK):
                    eid = flat_idx[t, k].item()
                    p = flat_val[t, k].item()
                    target[0][li, eid] += 1
                    target[1][li, eid] += p
            if last_layer:
                exe._tok_seen += flat_idx.size(0)

        return hook

    li = 0
    for name, module in exe.model_runner.model.named_modules():
        if PAT.search(name):
            module.register_forward_hook(mk(li))
            li += 1
    return li
llm.collective_rpc(_install)

# Process the single prompt
text = args.prompt
tokenizer = llm.get_tokenizer()
plen = len(tokenizer(text)["input_ids"])  # number of prompt tokens
def _reset(exe, plen):
    """Reset trace buffers before generation."""
    exe._trace_in.zero_()
    exe._trace_out.zero_()
    exe._sum_in.zero_()
    exe._sum_out.zero_()
    exe._tok_seen = 0
    exe._plen = plen
llm.collective_rpc(_reset, args=(plen,))

# Run generation
outputs = list(llm.generate([text], params))
gen_text = outputs[0].outputs[0].text

# Collect statistics from all workers
w = llm.collective_rpc(
    lambda e: dict(
        ti=e._trace_in.cpu().tolist(),
        to=e._trace_out.cpu().tolist(),
        si=e._sum_in.cpu().tolist(),
        so=e._sum_out.cpu().tolist(),
    ),
    timeout=None,
)[0]

# Compute average gating probability per expert
def compute_avg(counts, sums):
    cnt = np.array(counts, dtype=np.float32)
    s = np.array(sums, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(cnt > 0, s / cnt, 0.0)
    return avg.tolist()

avg_in = compute_avg(w["ti"], w["si"])
avg_out = compute_avg(w["to"], w["so"])

# Assemble JSON result
input_tokens = tokenizer(text)["input_ids"]
output_tokens = tokenizer(gen_text, add_special_tokens=False)["input_ids"]
result = {
    "model": args.model,
    "prompt": text,
    "input_tokens": input_tokens,
    "output_text": gen_text,
    "output_tokens": output_tokens,
    "num_layers": L,
    "num_experts": E,
    "top_k": TK,
    "input_expert_usage": w["ti"],  # [L, E] usage count for input tokens
    "input_avg_probs": avg_in,      # [L, E] average gate probability (input)
    "output_expert_usage": w["to"], # [L, E] usage count for generated tokens
    "output_avg_probs": avg_out,    # [L, E] average gate probability (output)
}

# Print & save results
json_str = json.dumps(result, ensure_ascii=False)
print(json_str)

save_path = pathlib.Path(getattr(args, "save", "moe_trace.json"))
save_path.write_text(json_str, encoding="utf-8")
print(f"✅ JSON saved to {save_path.resolve()}")
