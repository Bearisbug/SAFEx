<h1 style="text-align:center;">SAFEx: Analyzing Vulnerabilities of MoE-Based LLMs via Stable Safety-critical Expert Identification</h1>

<img width="3248" height="1792" alt="output-1" src="https://github.com/user-attachments/assets/5e94a21e-fd4d-46ae-ac1a-488560ce3fc4" />

[![Static Badge](https://img.shields.io/badge/2506.17368-white?style=flat&logo=arxiv&logoColor=%23B31B1B&color=white)](https://arxiv.org/abs/2506.17368)

> **The safety behavior of MoE LLMs is highly concentrated in a small set of “safety-critical experts”. Masking only a few of these experts can significantly reduce refusal rates on harmful prompts, while keeping general capabilities largely intact.**

The code here focuses on **observing and probing those experts** during inference, via two core scripts:

- `trace.py`: Trace **per-layer / per-expert** routing statistics for a single prompt
- `probe.py`: Probe **selected experts** to extract feature vectors for further analysis

---

## Model Setup

Both scripts assume a **vLLM-compatible MoE model directory**, for example:

* `Qwen3-30B-A3B`
* `Qwen1.5-MoE-A2.7B-Chat`
* `Mixtral-8x7B-Instruct-v0.1`
* `deepseek-moe-16b-chat`

and a `config.json` that provides:

* `num_hidden_layers`
* one of `num_experts`, `num_local_experts`, or `n_routed_experts`
* routing parameters like `num_experts_per_tok` or `router_top_k`

---

## `trace.py`

Given a prompt, this script:

* Loads the MoE model with vLLM (eager mode)
* Registers forward hooks on all MoE gate / router modules
* Records, for each layer and expert:

  * **Input phase (prompt tokens)**: usage counts & average gate probabilities
  * **Output phase (generated tokens)**: usage counts & average gate probabilities

It finally emits a JSON object containing:

* `model`: model path
* `prompt`: input text
* `input_tokens`, `output_tokens`, `output_text`
* `num_layers`, `num_experts`, `top_k`
* `input_expert_usage`, `output_expert_usage`: `[L, E]` integer matrices
* `input_avg_probs`, `output_avg_probs`: `[L, E]` float matrices

These statistics are directly useful for SAFEx-style analyses: ranking experts, building histograms, and performing stability-based expert selection.

### Example Usage

```bash
python trace.py \
  --model /path/to/moe-model \
  --prompt "Explain the difference between transformers and mixture-of-experts." \
  --max_new 128 \
  --dtype bfloat16 \
  --seed 42
```

This will:

* Print a single JSON line to stdout
* Save the same JSON to `moe_trace.json` in the current directory

---

## `probe.py`

`probe.py` is designed to **probe selected experts** at specified layers:

1. Installs forward-pre hooks on the chosen layers to cache hidden states **right before** the MoE FFN.
2. Runs a single normal forward pass (`llm.generate`) on a prompt, collecting those hidden states.
3. For each `(layer_idx, expert_idx)` pair:

   * Constructs a **one-hot router distribution**: all tokens are routed to that expert
   * Uses the relevant expert module (wrapped via `DeepSeekFusedAdapter` if necessary) to recompute FFN outputs
   * Averages over tokens to obtain a `[D]`-dimensional feature vector for that expert

The script writes a JSON dict:

* Key: `"layer-expert"` string (e.g., `"0-3"`)
* Value: the corresponding expert feature vector (list of length `D`)

### Example Usage

Edit the bottom of `probe.py`:

```python
if __name__ == "__main__":
    model_path = "/path/to/moe-model"
    prompt = "Explain the difference between transformers and mixture-of-experts."
    pairs = [(0, 3), (4, 7), (10, 1)]  # (layer_idx, expert_idx)

    probe_single_prompt(
        model_path=model_path,
        prompt=prompt,
        pairs=pairs,
        dtype="bfloat16",
        max_seq_len=4096,
        output_file="expert_features.json",
    )
```

Run:

```bash
python probe.py
```

You will get `expert_features.json`:

```json
{
  "0-3": [0.123, 0.456, ...],
  "4-7": [...],
  "10-1": [...]
}
```

## Citation

```bibtex
@misc{lai2025safexanalyzingvulnerabilitiesmoebased,
      title={SAFEx: Analyzing Vulnerabilities of MoE-Based LLMs via Stable Safety-critical Expert Identification}, 
      author={Zhenglin Lai and Mengyao Liao and Bingzhe Wu and Dong Xu and Zebin Zhao and Zhihang Yuan and Chao Fan and Jianqiang Li},
      year={2025},
      eprint={2506.17368},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.17368}, 
}
```
