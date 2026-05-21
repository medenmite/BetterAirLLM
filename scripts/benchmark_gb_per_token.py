"""Measure BetterAirLLM bytes moved per generated token.

This benchmark is intentionally byte-first. Tokens/sec alone can hide whether
the runtime is actually avoiding BetterAirLLM-style layer reads. The reported
GB/token values come from BetterAirLLM runtime counters around CPU shard loads and
host-to-device tensor materialization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
import tracemalloc


REPO_ROOT = Path(__file__).resolve().parents[1]
AIR_LLM_ROOT = REPO_ROOT / "air_llm"
if str(AIR_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(AIR_LLM_ROOT))

from betterairllm import BetterAirLLMBaseModel, AutoModel  # noqa: E402
from betterairllm.gpt_oss_mxfp4 import (  # noqa: E402
    FP4_VALUES,
    pack_dense_to_mxfp4_exact,
)
from betterairllm.moe_layout_probe import download_gpt_oss_one_expert_mxfp4_smoke, dry_run_gb_per_token_report  # noqa: E402
from betterairllm.selective_fused_moe import (  # noqa: E402
    FakeFusedMoEAdapter,
    GptOssSelectiveFusedMoEAdapter,
    build_fake_fused_expert_shards,
    build_fake_gpt_oss_expert_shards,
    fake_full_expert_bytes,
    fake_full_fused_moe_forward,
    fake_gpt_oss_full_expert_bytes,
    fake_gpt_oss_full_fused_moe_forward,
)


def current_peak_vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:
        return 0.0
    return 0.0


def build_model(args):
    kwargs = {
        "device": args.device,
        "max_seq_len": args.max_seq_len,
        "prefetching": not args.no_prefetching,
        "os_reserved_ram_mb": args.os_reserved_ram_mb,
        "abort_unsafe_context": args.abort_unsafe_context,
        "resume_split_build": args.resume_split_build,
        "lazy_build": args.lazy_build,
        "expert_materialization_mode": args.expert_materialization_mode,
    }
    if args.layer_shards_saving_path:
        kwargs["layer_shards_saving_path"] = args.layer_shards_saving_path
    if args.hf_token:
        kwargs["hf_token"] = args.hf_token
    if args.compression:
        kwargs["compression"] = args.compression

    if args.runtime == "base":
        return BetterAirLLMBaseModel(args.model, **kwargs)

    kwargs["moe_expert_cache_mb"] = args.moe_expert_cache_mb
    kwargs["moe_cpu_expert_cache_mb"] = args.moe_cpu_expert_cache_mb
    kwargs["moe_strict_streaming"] = args.moe_strict_streaming
    kwargs["moe_allow_dense_fallback"] = args.moe_allow_dense_fallback
    kwargs["mxfp4_execution"] = args.mxfp4_execution
    kwargs["mxfp4_device"] = args.mxfp4_device
    kwargs["expert_matmul_device"] = args.expert_matmul_device
    kwargs["hf_triton_module_cache_mb"] = args.hf_triton_module_cache_mb
    kwargs["expert_execution_mode"] = args.expert_execution_mode
    return AutoModel.from_pretrained(args.model, **kwargs)


def tensor_token_count(tensor):
    if hasattr(tensor, "shape"):
        return int(tensor.shape[-1])
    return len(tensor[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="Model repo id or local checkpoint path")
    parser.add_argument("--prompt", default="Explain mixture-of-experts inference in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime", choices=("auto", "base"), default="auto")
    parser.add_argument("--moe-expert-cache-mb", type=int, default=None)
    parser.add_argument("--moe-cpu-expert-cache-mb", type=int, default=0)
    parser.add_argument("--moe-strict-streaming", action="store_true", help="Fail instead of keeping full fused expert tensors when no selective adapter exists")
    parser.add_argument("--moe-allow-dense-fallback", action="store_true", help="Explicitly allow dense fused MoE fallback")
    parser.add_argument("--mxfp4-execution", choices=("reference_cuda", "hf_triton", "triton_fused"), default="hf_triton")
    parser.add_argument("--mxfp4-device", default="cuda")
    parser.add_argument("--expert-matmul-device", default="cuda")
    parser.add_argument("--hf-triton-module-cache-mb", type=int, default=0)
    parser.add_argument("--expert-execution-mode", choices=("per_token", "grouped_by_expert"), default="grouped_by_expert")
    parser.add_argument("--stream-mode", choices=("none", "compat", "text_iterator_streamer"), default="none")
    parser.add_argument("--os-reserved-ram-mb", type=int, default=8192, help="RAM reserved for the OS in safety estimates")
    parser.add_argument("--abort-unsafe-context", action="store_true", help="Abort when KV-cache safety estimate exceeds available RAM")
    parser.add_argument("--resume-split-build", action="store_true", help="Resume an interrupted incremental MoE split build")
    parser.add_argument("--lazy-build", action="store_true", help="Build missing MoE split shards lazily during runtime")
    parser.add_argument("--expert-materialization-mode", choices=("persisted_split", "direct_slice", "hybrid"), default="direct_slice")
    parser.add_argument("--layer-shards-saving-path")
    parser.add_argument("--compression", choices=("4bit", "8bit"))
    parser.add_argument("--hf-token")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--no-prefetching", action="store_true")
    parser.add_argument("--warm-cache", action="store_true", help="Keep any resident/RAM expert cache already present")
    parser.add_argument("--dry-run-layout", action="store_true", help="Report metadata-only GPT-OSS GB/token estimates and exit")
    parser.add_argument("--confirm-real-run", action="store_true", help="Required for real GPT-OSS benchmark/generation runs")
    parser.add_argument("--fake-fused-moe", action="store_true", help="Run synthetic fused dense vs selective MoE benchmark")
    parser.add_argument("--fake-gpt-oss-moe", action="store_true", help="Run synthetic GPT-OSS fused dense vs selective MoE benchmark")
    parser.add_argument("--fake-gpt-oss-mxfp4", action="store_true", help="Run synthetic GPT-OSS MXFP4 packed selective benchmark")
    parser.add_argument("--real-gpt-oss-one-expert-smoke", action="store_true", help="Download and dequantize one real GPT-OSS 20B MXFP4 expert")
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--confirm-download", action="store_true")
    parser.add_argument("--fake-hidden-dim", type=int, default=64)
    parser.add_argument("--fake-intermediate-dim", type=int, default=128)
    parser.add_argument("--fake-num-experts", type=int, default=16)
    parser.add_argument("--fake-top-k", type=int, default=4)
    parser.add_argument("--fake-tokens", type=int, default=4)
    args = parser.parse_args()

    if args.fake_fused_moe:
        return run_fake_fused_moe_benchmark(args)
    if args.fake_gpt_oss_moe:
        return run_fake_gpt_oss_moe_benchmark(args)
    if args.fake_gpt_oss_mxfp4:
        return run_fake_gpt_oss_mxfp4_benchmark(args)

    if not args.model:
        parser.error("model is required unless a fake benchmark is used")

    if args.real_gpt_oss_one_expert_smoke:
        if not args.confirm_download:
            raise SystemExit("--real-gpt-oss-one-expert-smoke requires --confirm-download")
        started = time.perf_counter()
        report = download_gpt_oss_one_expert_mxfp4_smoke(
            args.model,
            layer_id=args.layer_id,
            expert_id=args.expert_id,
            cache_dir=args.hf_cache_dir,
            token=args.hf_token,
        )
        report["seconds"] = time.perf_counter() - started
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.dry_run_layout:
        report = dry_run_gb_per_token_report(
            args.model,
            cache_dir=args.hf_cache_dir,
            token=args.hf_token,
            vram_cache_mb=args.moe_expert_cache_mb,
            cpu_cache_mb=args.moe_cpu_expert_cache_mb,
            os_reserved_ram_mb=args.os_reserved_ram_mb,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    needs_real_run_gate = "gpt-oss" in args.model.lower()
    preflight_report = None
    if needs_real_run_gate:
        preflight_report = dry_run_gb_per_token_report(
            args.model,
            cache_dir=args.hf_cache_dir,
            token=args.hf_token,
            vram_cache_mb=args.moe_expert_cache_mb,
            cpu_cache_mb=args.moe_cpu_expert_cache_mb,
            os_reserved_ram_mb=args.os_reserved_ram_mb,
        )
        print(json.dumps({"preflight_layout_estimate": preflight_report}, indent=2, sort_keys=True))
        if not args.confirm_real_run:
            raise SystemExit(
                "Refusing real GPT-OSS run without --confirm-real-run. "
                "Use --dry-run-layout for metadata-only estimates."
            )
        if args.moe_strict_streaming and not preflight_report.get("strict_selective_runtime_can_enable") and not args.moe_allow_dense_fallback:
            raise SystemExit(
                "Refusing strict GPT-OSS run before model loading because the current real checkpoint layout "
                f"is not execution-ready: {preflight_report.get('blockers')}"
            )

    model = build_model(args)
    tokenizer = model.tokenizer
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    input_tokens = tensor_token_count(input_ids)

    if not args.warm_cache and hasattr(model, "clear_runtime_caches"):
        model.clear_runtime_caches()
    else:
        model.reset_runtime_stats()
    started = time.perf_counter()
    output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
    elapsed = time.perf_counter() - started

    output_tokens = tensor_token_count(output)
    generated_tokens = max(0, output_tokens - input_tokens)
    stats = model.runtime_stats()

    cpu_load_gb = stats.get("cpu_load_bytes", 0) / 1024 ** 3
    device_load_gb = stats.get("device_load_bytes", 0) / 1024 ** 3
    denom = max(1, generated_tokens)

    report = {
        "model": args.model,
        "runtime": args.runtime,
        "stream_mode": args.stream_mode,
        "mxfp4_execution": args.mxfp4_execution,
        "expert_execution_mode": args.expert_execution_mode,
        "hf_triton_module_cache_mb": args.hf_triton_module_cache_mb,
        "prompt_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "time_to_first_token_seconds": elapsed if generated_tokens == 1 else None,
        "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else 0.0,
        "peak_vram_mb": current_peak_vram_mb(),
        "cpu_load_gb": cpu_load_gb,
        "device_load_gb": device_load_gb,
        "cpu_load_gb_per_generated_token": cpu_load_gb / denom,
        "device_load_gb_per_generated_token": device_load_gb / denom,
        "mxfp4": {
            "packed_expert_gb_per_generated_token": stats.get("packed_expert_bytes_loaded", 0) / 1024 ** 3 / denom,
            "dequant_temp_gb_per_generated_token": stats.get("dequantized_temporary_bytes", 0) / 1024 ** 3 / denom,
            "mxfp4_dequant_seconds": stats.get("mxfp4_dequant_seconds", 0.0),
            "hf_triton_kernel_seconds": stats.get("hf_triton_kernel_seconds", 0.0),
            "hf_triton_module_cache_hits": stats.get("hf_triton_module_cache_hits", 0),
            "hf_triton_module_cache_misses": stats.get("hf_triton_module_cache_misses", 0),
            "hf_triton_fallbacks": stats.get("hf_triton_fallbacks", 0),
            "hf_triton_last_error": stats.get("hf_triton_last_error", ""),
        },
        "runtime_stats": stats,
    }
    if needs_real_run_gate and args.max_new_tokens == 1 and preflight_report is not None:
        dense_fallback_gb = preflight_report["dense_gb_per_token"]
        packed_gb = stats.get("packed_expert_bytes_loaded", 0) / 1024 ** 3 / denom
        dequant_gb = stats.get("dequantized_temporary_bytes", 0) / 1024 ** 3 / denom
        avoided_gb = preflight_report["bytes_avoided"] / 1024 ** 3 / denom
        report["betterairllm_gpt_oss_20b_one_token_smoke"] = {
            "packed_gb_per_token": packed_gb,
            "dequant_temp_gb_per_token": dequant_gb,
            "dense_fallback_gb_per_token_estimate": dense_fallback_gb,
            "avoided_gb_per_token": avoided_gb,
            "speed_tokens_per_second": generated_tokens / elapsed if elapsed > 0 else 0.0,
            "correctness_status": "runtime_completed",
            "strict_mode": bool(args.moe_strict_streaming),
            "dense_fallback_used": bool(stats.get("requires_dense_fallback")),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


def run_fake_fused_moe_benchmark(args):
    import torch

    torch.manual_seed(123)
    router = torch.nn.Linear(args.fake_hidden_dim, args.fake_num_experts, bias=False)
    gate_up_proj_weight = torch.randn(
        args.fake_num_experts,
        2 * args.fake_intermediate_dim,
        args.fake_hidden_dim,
    ) * 0.02
    down_proj_weight = torch.randn(
        args.fake_num_experts,
        args.fake_hidden_dim,
        args.fake_intermediate_dim,
    ) * 0.02
    hidden_states = torch.randn(1, args.fake_tokens, args.fake_hidden_dim)
    shards = build_fake_fused_expert_shards(gate_up_proj_weight, down_proj_weight)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    dense_started = time.perf_counter()
    dense_output = fake_full_fused_moe_forward(
        hidden_states,
        router,
        gate_up_proj_weight,
        down_proj_weight,
        args.fake_top_k,
    )
    dense_elapsed = time.perf_counter() - dense_started
    dense_ram_current, dense_ram_peak = tracemalloc.get_traced_memory()
    dense_peak_vram_mb = current_peak_vram_mb()
    tracemalloc.stop()

    adapter = FakeFusedMoEAdapter(
        router=router,
        expert_loader=lambda expert_id: shards[int(expert_id)],
        num_experts=args.fake_num_experts,
        top_k=args.fake_top_k,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    selective_started = time.perf_counter()
    selective_output = adapter.forward(hidden_states)
    selective_elapsed = time.perf_counter() - selective_started
    selective_ram_current, selective_ram_peak = tracemalloc.get_traced_memory()
    selective_peak_vram_mb = current_peak_vram_mb()
    tracemalloc.stop()

    torch.testing.assert_close(selective_output, dense_output, rtol=1e-5, atol=1e-6)

    full_expert_bytes = fake_full_expert_bytes(gate_up_proj_weight, down_proj_weight)
    dense_bytes = full_expert_bytes
    selective_bytes = adapter.stats["expert_load_bytes"]
    denom = max(1, args.fake_tokens)

    report = {
        "benchmark": "fake_fused_moe",
        "correctness": "selective_matches_dense",
        "fake_hidden_dim": args.fake_hidden_dim,
        "fake_intermediate_dim": args.fake_intermediate_dim,
        "fake_num_experts": args.fake_num_experts,
        "fake_top_k": args.fake_top_k,
        "fake_tokens": args.fake_tokens,
        "dense": {
            "bytes_loaded": dense_bytes,
            "gb_per_token": dense_bytes / 1024 ** 3 / denom,
            "tokens_per_second": args.fake_tokens / dense_elapsed if dense_elapsed > 0 else 0.0,
            "peak_ram_mb": dense_ram_peak / 1024 ** 2,
            "peak_vram_mb": dense_peak_vram_mb,
        },
        "selective": {
            "expert_bytes_loaded": selective_bytes,
            "total_gb_per_token": selective_bytes / 1024 ** 3 / denom,
            "selected_experts_per_token": adapter.stats["selected_expert_slots"] / denom,
            "unique_selected_experts": adapter.stats["unique_selected_experts"],
            "full_expert_bytes_avoided": dense_bytes - selective_bytes,
            "tokens_per_second": args.fake_tokens / selective_elapsed if selective_elapsed > 0 else 0.0,
            "peak_ram_mb": selective_ram_peak / 1024 ** 2,
            "peak_vram_mb": selective_peak_vram_mb,
            "cache_hit_rate": 0.0,
            "adapter_stats": adapter.stats,
        },
        "acceptance": {
            "selective_lower_gb_per_token": selective_bytes < dense_bytes,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


class _FakeGptOssRouter:
    def __init__(self, weight, bias, top_k):
        self.weight = weight
        self.bias = bias
        self.top_k = top_k

    def __call__(self, hidden_states):
        import torch
        import torch.nn.functional as F

        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_scores = torch.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        return router_logits, router_scores, router_indices


def run_fake_gpt_oss_moe_benchmark(args):
    import torch

    torch.manual_seed(120)
    router_weight = torch.randn(args.fake_num_experts, args.fake_hidden_dim) * 0.1
    router_bias = torch.randn(args.fake_num_experts) * 0.02
    router = _FakeGptOssRouter(router_weight, router_bias, args.fake_top_k)
    gate_up_proj = torch.randn(
        args.fake_num_experts,
        args.fake_hidden_dim,
        2 * args.fake_intermediate_dim,
    ) * 0.02
    gate_up_proj_bias = torch.randn(args.fake_num_experts, 2 * args.fake_intermediate_dim) * 0.01
    down_proj = torch.randn(
        args.fake_num_experts,
        args.fake_intermediate_dim,
        args.fake_hidden_dim,
    ) * 0.02
    down_proj_bias = torch.randn(args.fake_num_experts, args.fake_hidden_dim) * 0.01
    hidden_states = torch.randn(1, args.fake_tokens, args.fake_hidden_dim)
    shards = build_fake_gpt_oss_expert_shards(gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    dense_started = time.perf_counter()
    dense_output = fake_gpt_oss_full_fused_moe_forward(
        hidden_states,
        router,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
    )
    dense_elapsed = time.perf_counter() - dense_started
    _, dense_ram_peak = tracemalloc.get_traced_memory()
    dense_peak_vram_mb = current_peak_vram_mb()
    tracemalloc.stop()

    adapter = GptOssSelectiveFusedMoEAdapter(
        router=router,
        expert_loader=lambda expert_id: shards[int(expert_id)],
        num_experts=args.fake_num_experts,
        top_k=args.fake_top_k,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    selective_started = time.perf_counter()
    selective_output = adapter.forward(hidden_states)
    selective_elapsed = time.perf_counter() - selective_started
    _, selective_ram_peak = tracemalloc.get_traced_memory()
    selective_peak_vram_mb = current_peak_vram_mb()
    tracemalloc.stop()

    torch.testing.assert_close(selective_output, dense_output, rtol=1e-5, atol=1e-6)

    dense_bytes = fake_gpt_oss_full_expert_bytes(gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias)
    selective_bytes = adapter.stats["expert_load_bytes"]
    denom = max(1, args.fake_tokens)
    report = {
        "benchmark": "fake_gpt_oss_moe",
        "correctness": "selective_matches_dense",
        "fake_hidden_dim": args.fake_hidden_dim,
        "fake_intermediate_dim": args.fake_intermediate_dim,
        "fake_num_experts": args.fake_num_experts,
        "fake_top_k": args.fake_top_k,
        "fake_tokens": args.fake_tokens,
        "dense": {
            "bytes_loaded": dense_bytes,
            "gb_per_token": dense_bytes / 1024 ** 3 / denom,
            "tokens_per_second": args.fake_tokens / dense_elapsed if dense_elapsed > 0 else 0.0,
            "peak_ram_mb": dense_ram_peak / 1024 ** 2,
            "peak_vram_mb": dense_peak_vram_mb,
        },
        "selective": {
            "expert_bytes_loaded": selective_bytes,
            "total_gb_per_token": selective_bytes / 1024 ** 3 / denom,
            "selected_experts_per_token": adapter.stats["selected_expert_slots"] / denom,
            "top_k": args.fake_top_k,
            "unique_selected_experts": adapter.stats["unique_selected_experts"],
            "full_expert_bytes_avoided": dense_bytes - selective_bytes,
            "tokens_per_second": args.fake_tokens / selective_elapsed if selective_elapsed > 0 else 0.0,
            "peak_ram_mb": selective_ram_peak / 1024 ** 2,
            "peak_vram_mb": selective_peak_vram_mb,
            "cache_hit_rate": 0.0,
            "adapter_stats": adapter.stats,
        },
        "acceptance": {
            "selective_lower_gb_per_token": selective_bytes < dense_bytes,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _representable_dense(shape, offset=0):
    import torch

    values = torch.tensor(FP4_VALUES, dtype=torch.float32)
    total = 1
    for dim in shape:
        total *= dim
    return (values[(torch.arange(total) + offset) % len(values)].reshape(shape) * 0.125).contiguous()


def run_fake_gpt_oss_mxfp4_benchmark(args):
    import torch

    torch.manual_seed(404)
    hidden_dim = max(32, args.fake_hidden_dim)
    intermediate_dim = max(32, args.fake_intermediate_dim)
    hidden_dim = ((hidden_dim + 31) // 32) * 32
    intermediate_dim = ((intermediate_dim + 31) // 32) * 32
    num_experts = args.fake_num_experts
    top_k = min(args.fake_top_k, num_experts)

    gate_up_dense = _representable_dense((num_experts, hidden_dim, 2 * intermediate_dim), offset=2)
    down_dense = _representable_dense((num_experts, intermediate_dim, hidden_dim), offset=5)
    gate_up_blocks, gate_up_scales = pack_dense_to_mxfp4_exact(gate_up_dense, scale_exponent=-3)
    down_blocks, down_scales = pack_dense_to_mxfp4_exact(down_dense, scale_exponent=-3)
    gate_up_bias = torch.randn(num_experts, 2 * intermediate_dim) * 0.01
    down_bias = torch.randn(num_experts, hidden_dim) * 0.01
    hidden_states = torch.randn(1, args.fake_tokens, hidden_dim) * 0.1
    router = _FakeGptOssRouter(torch.randn(num_experts, hidden_dim) * 0.1, torch.randn(num_experts) * 0.02, top_k)

    dense_shards = build_fake_gpt_oss_expert_shards(gate_up_dense, gate_up_bias, down_dense, down_bias)
    packed_shards = {}
    for expert_id in range(num_experts):
        packed_shards[expert_id] = {
            "gate_up_proj_blocks": gate_up_blocks[expert_id].contiguous(),
            "gate_up_proj_scales": gate_up_scales[expert_id].contiguous(),
            "gate_up_proj_bias": gate_up_bias[expert_id].contiguous(),
            "down_proj_blocks": down_blocks[expert_id].contiguous(),
            "down_proj_scales": down_scales[expert_id].contiguous(),
            "down_proj_bias": down_bias[expert_id].contiguous(),
        }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    dense_started = time.perf_counter()
    dense_output = fake_gpt_oss_full_fused_moe_forward(
        hidden_states,
        router,
        gate_up_dense,
        gate_up_bias,
        down_dense,
        down_bias,
    )
    dense_elapsed = time.perf_counter() - dense_started
    _, dense_ram_peak = tracemalloc.get_traced_memory()
    dense_peak_vram_mb = current_peak_vram_mb()
    tracemalloc.stop()

    adapter = GptOssSelectiveFusedMoEAdapter(
        router=router,
        expert_loader=lambda expert_id: packed_shards[int(expert_id)],
        num_experts=num_experts,
        top_k=top_k,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    selective_started = time.perf_counter()
    selective_output = adapter.forward(hidden_states)
    selective_elapsed = time.perf_counter() - selective_started
    _, selective_ram_peak = tracemalloc.get_traced_memory()
    selective_peak_vram_mb = current_peak_vram_mb()
    tracemalloc.stop()
    torch.testing.assert_close(selective_output, dense_output, rtol=2e-2, atol=2e-2)

    full_dense_bytes = fake_gpt_oss_full_expert_bytes(gate_up_dense, gate_up_bias, down_dense, down_bias)
    packed_expert_bytes_loaded = adapter.stats["expert_load_bytes"]
    _, _, selected_indices = router(hidden_states.reshape(-1, hidden_dim))
    unique_selected = sorted(int(item) for item in torch.unique(selected_indices).tolist())
    dequantized_temp_bytes = sum(
        gate_up_dense[expert_id].numel() * gate_up_dense.element_size()
        + down_dense[expert_id].numel() * down_dense.element_size()
        + gate_up_bias[expert_id].numel() * gate_up_bias.element_size()
        + down_bias[expert_id].numel() * down_bias.element_size()
        for expert_id in unique_selected
    )
    denom = max(1, args.fake_tokens)
    report = {
        "benchmark": "fake_gpt_oss_mxfp4",
        "correctness": "selective_mxfp4_matches_dense_reference",
        "top_k": top_k,
        "selected_experts_per_token": adapter.stats["selected_expert_slots"] / denom,
        "packed_expert_bytes_loaded": packed_expert_bytes_loaded,
        "dequantized_temporary_bytes": dequantized_temp_bytes,
        "dense_fallback_bytes": full_dense_bytes,
        "bytes_avoided": full_dense_bytes - packed_expert_bytes_loaded,
        "gb_per_token": packed_expert_bytes_loaded / 1024 ** 3 / denom,
        "cache_hit_rate": 0.0,
        "seconds_per_token": selective_elapsed / denom,
        "dense": {
            "bytes_loaded": full_dense_bytes,
            "tokens_per_second": args.fake_tokens / dense_elapsed if dense_elapsed > 0 else 0.0,
            "peak_ram_mb": dense_ram_peak / 1024 ** 2,
            "peak_vram_mb": dense_peak_vram_mb,
        },
        "selective": {
            "tokens_per_second": args.fake_tokens / selective_elapsed if selective_elapsed > 0 else 0.0,
            "peak_ram_mb": selective_ram_peak / 1024 ** 2,
            "peak_vram_mb": selective_peak_vram_mb,
            "adapter_stats": adapter.stats,
            "unique_selected_experts": unique_selected,
        },
        "acceptance": {
            "selective_lower_gb_per_token": packed_expert_bytes_loaded < full_dense_bytes,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
