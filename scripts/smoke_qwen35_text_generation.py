"""Qwen3.5/Qwen3.6 MoE strict text-generation smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AIR_LLM_ROOT = REPO_ROOT / "air_llm"
if str(AIR_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(AIR_LLM_ROOT))

import torch  # noqa: E402
from betterairllm import AutoModel  # noqa: E402


def _dtype_from_name(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


def _build_prompt(model, text: str):
    messages = [{"role": "user", "content": text}]
    tokenizer = model.tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Qwen3.5/Qwen3.6 MoE repo id or local checkpoint directory")
    parser.add_argument("--prompt", default="Write one short sentence about local LLMs.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dense-device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-vram-mb", type=int, default=7600)
    parser.add_argument("--vram-cache-mode", default="packed_experts")
    parser.add_argument("--vram-cache-mb", type=int, default=4000)
    parser.add_argument("--cpu-cache-mb", type=int, default=12000)
    parser.add_argument("--dense-expert-cache-min-requests", type=int, default=1)
    parser.add_argument("--layer-cleanup-interval", type=int, default=8)
    parser.add_argument("--expert-execution-mode", default="grouped_by_expert")
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable Qwen DynamicCache if cache validation fails on your environment")
    parser.add_argument("--hf-token")
    args = parser.parse_args()

    model = AutoModel.from_pretrained(
        args.model,
        device=args.device,
        dense_device=args.dense_device,
        dtype=_dtype_from_name(args.dtype),
        max_seq_len=args.max_seq_len,
        hf_token=args.hf_token,
        moe_strict_streaming=True,
        moe_allow_dense_fallback=False,
        moe_expert_cache_mb=1024,
        moe_cpu_expert_cache_mb=args.cpu_cache_mb,
        vram_cache_mode=args.vram_cache_mode,
        vram_cache_mb=args.vram_cache_mb,
        dense_expert_cache_min_requests=args.dense_expert_cache_min_requests,
        max_vram_mb=args.max_vram_mb,
        abort_unsafe_context=True,
        resume_split_build=True,
        lazy_build=True,
        expert_materialization_mode="direct_slice",
        enable_qwen35_batch_direct_slice=True,
        layer_cleanup_interval=args.layer_cleanup_interval,
        expert_execution_mode=args.expert_execution_mode,
        keep_nontransformer_resident=True,
        quiet_progress=False,
    )

    prompt = _build_prompt(model, args.prompt)
    tokens = model.tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=args.max_seq_len,
        padding=False,
    )
    input_ids = tokens["input_ids"]
    if str(args.device).startswith("cuda"):
        input_ids = input_ids.cuda()

    started = time.perf_counter()
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=not args.no_kv_cache,
        return_dict_in_generate=True,
    )
    elapsed = time.perf_counter() - started
    output_ids = output.sequences[0]
    new_ids = output_ids[input_ids.shape[-1]:]
    text = model.tokenizer.decode(new_ids.detach().cpu(), skip_special_tokens=True)
    runtime_stats = model.runtime_stats() if hasattr(model, "runtime_stats") else {}

    print(text.strip())
    print(json.dumps({
        "generated_tokens": int(new_ids.numel()),
        "elapsed_seconds": elapsed,
        "seconds_per_token": elapsed / max(1, int(new_ids.numel())),
        "dense_fallback_used": bool(runtime_stats.get("requires_dense_fallback", False)),
        "strict_full_fused_tensor_loaded": bool(runtime_stats.get("dense_contains_full_fused_experts", False)),
        "unused_expert_builds_this_run": int(runtime_stats.get("unused_expert_builds_this_run", 0)),
        "runtime": runtime_stats,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
