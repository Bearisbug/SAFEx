#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoE Expert Probe (Single Prompt Version)

Given a MoE model loaded by vLLM and a single text prompt, this script:
- installs hooks on selected MoE layers to capture their input hidden states;
- for each specified (layer, expert) pair, re-runs that expert with a
  one-hot router distribution;
- averages token-wise outputs to obtain one feature vector per expert;
- saves all feature vectors to a JSON file.

Supports architectures such as Qwen, Mixtral, DeepSeek-MoE, Qwen3-MoE, etc.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from vllm import LLM, SamplingParams
from vllm.forward_context import set_forward_context

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


class DeepSeekFusedAdapter(nn.Module):
    """Wrap DeepSeek-MoE experts (ModuleList of MLPs) into a single callable layer."""

    def __init__(self, experts: nn.ModuleList) -> None:
        super().__init__()
        self.experts = experts

    def forward(self, hidden: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """
        hidden: [T, D]
        router_logits: [T, E]
        """
        probs = router_logits.softmax(dim=-1)                # [T, E]
        expert_outs = torch.stack([e(hidden) for e in self.experts], dim=1)  # [T, E, D]
        return (expert_outs * probs.unsqueeze(-1)).sum(dim=1)               # [T, D]


def get_num_experts(experts_obj) -> int:
    """Return number of experts for Qwen/Mixtral/DeepSeek-style MoE blocks."""
    if isinstance(experts_obj, nn.ModuleList):
        return len(experts_obj)
    for attr in ["w13_weight", "w1", "w3", "w1.weight"]:
        if hasattr(experts_obj, attr):
            return getattr(experts_obj, attr).shape[0]
    return next(experts_obj.parameters()).shape[0]


def install_hooks(executor, layer_ids: List[int]) -> None:
    """
    Install forward_pre_hooks on the given layers to capture hidden states
    right before MoE experts are applied.
    """
    executor._hidden_states = {}
    model = getattr(executor.model_runner.model, "model", executor.model_runner.model)
    layers = model.layers

    if not hasattr(executor, "_hook_handles"):
        executor._hook_handles = []

    for layer_idx in layer_ids:
        layer = layers[layer_idx]
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            target_module = layer.mlp
        else:
            target_module = layer.block_sparse_moe

        def make_hook(idx: int):
            def _hook(_module, inputs):
                # inputs[0]: [T, D], we move to CPU for safety
                executor._hidden_states[idx] = inputs[0].detach().cpu()
            return _hook

        handle = target_module.register_forward_pre_hook(make_hook(layer_idx))
        executor._hook_handles.append(handle)


def list_all_experts(executor) -> List[Tuple[int, int]]:
    """
    Enumerate all (layer, expert) pairs from the underlying MoE model.
    """
    pairs: List[Tuple[int, int]] = []
    model = getattr(executor.model_runner.model, "model", executor.model_runner.model)
    layers = model.layers

    for lid, layer in enumerate(layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            experts = layer.mlp.experts
        elif hasattr(layer, "block_sparse_moe"):
            experts = layer.block_sparse_moe.experts
        else:
            continue

        num_exp = get_num_experts(experts)
        for eid in range(num_exp):
            pairs.append((lid, eid))

    return pairs


def compute_expert_features(
    executor,
    layer_idx: int,
    expert_ids: List[int],
) -> Dict[int, List[float]]:
    """
    For a specific layer and a list of expert IDs:
    - take the cached hidden states before that layer (shape [T, D]),
    - force the router to send ALL tokens to each expert (one-hot routing),
    - run the experts, and average outputs across tokens to get a [D]-dim feature.
    """
    hidden = executor._hidden_states[layer_idx]  # [T, D] on CPU

    model = getattr(executor.model_runner.model, "model", executor.model_runner.model)
    layer = model.layers[layer_idx]

    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        experts_raw = layer.mlp.experts
    else:
        experts_raw = layer.block_sparse_moe.experts

    if isinstance(experts_raw, nn.ModuleList):
        experts_module: nn.Module = DeepSeekFusedAdapter(experts_raw)
    else:
        experts_module = experts_raw

    device = next(experts_raw.parameters()).device
    hidden = hidden.to(device)
    num_experts = get_num_experts(experts_raw)
    T, _ = hidden.shape

    results: Dict[int, List[float]] = {}

    vllm_config = executor.vllm_config
    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        for eid in expert_ids:
            router_logits = torch.full(
                (T, num_experts),
                -1e9,
                dtype=hidden.dtype,
                device=device,
            )
            router_logits[:, eid] = 0.0  

            with torch.no_grad():
                out = experts_module(hidden, router_logits)  # [T, D]

            results[eid] = out.mean(dim=0).cpu().tolist()

    return results


def probe_single_prompt(
    model_path: str,
    prompt: str,
    pairs: List[Tuple[int, int]],
    dtype: str = "bfloat16",
    max_seq_len: int = 4096,
    output_file: str = "moe_probe_single.json",
) -> Dict[str, List[float]]:
    """
    Run MoE expert probing on a single prompt.

    Args:
        model_path: Path to the MoE model (HF/vLLM compatible directory).
        prompt: Text prompt to run through the model.
        pairs: List of (layer_index, expert_index) pairs to probe.
        dtype: vLLM dtype, e.g., "bfloat16" or "float16".
        max_seq_len: Max model length used by vLLM.
        output_file: Where to save the resulting JSON.

    Returns:
        A dict mapping "layer-expert" string keys to expert feature vectors.
    """
    print(f"[Init] Loading model from {model_path} ...")
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        trust_remote_code=True,
        dtype=dtype,
        max_model_len=max_seq_len,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    all_pairs = [tuple(p) for p in llm.collective_rpc(list_all_experts)[0]]
    available = set(all_pairs)
    if not set(pairs).issubset(available):
        raise ValueError(f"Invalid pairs specified: {set(pairs) - available}")

    layers = sorted({l for l, _ in pairs})
    llm.collective_rpc(lambda exe: install_hooks(exe, layers))

    params = SamplingParams(max_tokens=1, temperature=0.0)
    _ = list(llm.generate([prompt], params))  # trigger one forward pass

    feature_dict: Dict[str, List[float]] = {}
    for l in layers:
        expert_ids = [e for ll, e in pairs if ll == l]
        feats = llm.collective_rpc(
            lambda exe, li=l, eids=expert_ids: compute_expert_features(exe, li, eids)
        )[0]
        for e, vec in feats.items():
            key = f"{l}-{e}"
            feature_dict[key] = vec

    Path(output_file).write_text(
        json.dumps(feature_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Done] Saved expert features to {output_file}")
    return feature_dict


if __name__ == "__main__":
    # Example usage: modify these values as needed.
    model_path = ""
    prompt = "Explain the difference between transformers and mixture-of-experts."
    pairs = [(0, 3), (4, 7), (10, 1)]

    probe_single_prompt(
        model_path=model_path,
        prompt=prompt,
        pairs=pairs,
        dtype="bfloat16",
        max_seq_len=4096,
        output_file="expert_features.json",
    )
