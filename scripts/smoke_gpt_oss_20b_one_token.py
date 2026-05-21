"""Safe staged runtime smoke test for BetterAirLLM GPT-OSS 20B.

This script is a runtime smoke test, not an output quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
import copy
import sys
import threading
import time
import tracemalloc
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AIR_LLM_ROOT = REPO_ROOT / "air_llm"
if str(AIR_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(AIR_LLM_ROOT))

import torch  # noqa: E402

from betterairllm import AutoModel  # noqa: E402
from betterairllm.betterairllm_base import BetterAirLLMBaseModel  # noqa: E402
from betterairllm.moe_layout_probe import (  # noqa: E402
    download_gpt_oss_one_expert_mxfp4_smoke,
    dry_run_gb_per_token_report,
    load_config,
    load_safetensors_index,
)


def _dtype_from_name(name):
    lowered = (name or "").lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


def _kv_cache_estimate_from_config(config, max_seq_len, dtype):
    hidden_size = config.get("hidden_size")
    num_layers = config.get("num_hidden_layers")
    num_heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads", num_heads)
    if not hidden_size or not num_layers or not num_heads or not kv_heads:
        return None
    head_dim = config.get("head_dim") or hidden_size // num_heads
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    return int(num_layers) * 2 * int(max_seq_len) * int(kv_heads) * int(head_dim) * dtype_bytes


def _cuda_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _runtime_environment(args):
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    cuda_total_bytes = None
    cuda_free_bytes = None
    if cuda_available:
        device = torch.device(args.device if str(args.device).startswith("cuda") else "cuda:0")
        gpu_name = torch.cuda.get_device_name(device)
        cuda_free_bytes, cuda_total_bytes = torch.cuda.mem_get_info(device)
    dense_device = args.dense_device or args.device
    mxfp4_device = args.mxfp4_device or dense_device
    expert_matmul_device = args.expert_matmul_device or mxfp4_device
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "cuda_total_bytes": cuda_total_bytes,
        "cuda_free_bytes": cuda_free_bytes,
        "requested_device": args.device,
        "dense_device": dense_device,
        "mxfp4_device": mxfp4_device,
        "expert_matmul_device": expert_matmul_device,
        "max_vram_mb": args.max_vram_mb,
    }


def _validate_device_args(args):
    requested = [args.device, args.dense_device, args.mxfp4_device, args.expert_matmul_device]
    wants_cuda = any(str(item).startswith("cuda") for item in requested if item)
    if wants_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false in this Python environment.")
    if args.max_vram_mb is not None and args.max_vram_mb <= 0:
        raise SystemExit("--max-vram-mb must be positive")


def _is_local_model_dir(model):
    return Path(str(model)).expanduser().is_dir()


def _is_gpt_oss_config(config):
    architectures = config.get("architectures") or []
    return (
        str(config.get("model_type", "")).lower() == "gpt_oss"
        or any("gptossforcausallm" in str(item).replace("_", "").lower() for item in architectures)
        or any("gpt-oss" in str(item).lower() for item in architectures)
    )


def _is_probably_120b(args, config):
    model_lower = str(args.model).lower()
    if "gpt-oss-120b" in model_lower or "120b" in model_lower:
        return True
    num_layers = int(config.get("num_hidden_layers") or 0)
    num_experts = int(config.get("num_local_experts") or config.get("num_experts") or 0)
    return num_layers >= 36 and num_experts >= 128


class MemorySampler:
    def __init__(self):
        self.peak_rss = None
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._process = psutil.Process()
        except Exception:
            self._process = None

    def __enter__(self):
        tracemalloc.start()
        if self._process is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_tracemalloc_bytes = peak

    def _sample(self):
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
                self.peak_rss = rss if self.peak_rss is None else max(self.peak_rss, rss)
            except Exception:
                pass
            time.sleep(0.05)


def _build_preflight(args):
    dtype = _dtype_from_name(args.dtype)
    config = load_config(args.model, cache_dir=args.hf_cache_dir, token=args.hf_token)
    report = dry_run_gb_per_token_report(
        args.model,
        cache_dir=args.hf_cache_dir,
        token=args.hf_token,
        vram_cache_mb=args.moe_expert_cache_mb,
        cpu_cache_mb=args.moe_cpu_expert_cache_mb,
        os_reserved_ram_mb=args.os_reserved_ram_mb,
    )
    kv_estimate = _kv_cache_estimate_from_config(config, args.max_seq_len, dtype)
    available_ram = BetterAirLLMBaseModel._available_ram_bytes()
    safe_ram = None if available_ram is None else max(0, available_ram - args.os_reserved_ram_mb * 1024 ** 2)

    estimates = report["probe"]["estimates"]
    probe = report["probe"]
    per_expert_bytes = int(estimates.get("per_expert_bytes") or 0)
    num_experts = int(probe.get("num_experts") or 0)
    top_k_for_prompt = int(report.get("top_k") or probe.get("experts_per_token") or 1)
    max_prompt_unique_experts = min(num_experts, int(args.max_seq_len) * max(1, top_k_for_prompt)) if num_experts else 0
    largest_dequant_temp = max(
        int(estimates["selected_expert_bytes_per_layer_token"]),
        per_expert_bytes * max_prompt_unique_experts,
    )
    ram_needed = (kv_estimate or 0) + largest_dequant_temp + args.moe_cpu_expert_cache_mb * 1024 ** 2
    vram_needed = largest_dequant_temp + (args.moe_expert_cache_mb or 0) * 1024 ** 2
    cuda_free = None
    cuda_total = None
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        cuda_free, cuda_total = torch.cuda.mem_get_info(torch.device(args.device))

    guard_failures = []
    if safe_ram is not None and ram_needed > safe_ram:
        guard_failures.append(
            f"estimated RAM need {ram_needed / 1024 ** 3:.2f}GB exceeds safe available RAM {safe_ram / 1024 ** 3:.2f}GB"
        )
    if cuda_free is not None and vram_needed > max(0, cuda_free - 512 * 1024 ** 2):
        guard_failures.append(
            f"estimated VRAM/cache need {vram_needed / 1024 ** 3:.2f}GB exceeds free VRAM safety budget {(cuda_free - 512 * 1024 ** 2) / 1024 ** 3:.2f}GB"
        )
    if args.max_vram_mb is not None and vram_needed > int(args.max_vram_mb) * 1024 ** 2:
        guard_failures.append(
            f"estimated VRAM/cache need {vram_needed / 1024 ** 3:.2f}GB exceeds --max-vram-mb budget {args.max_vram_mb / 1024:.2f}GB"
        )

    expected_layers = int(report["probe"].get("num_hidden_layers") or 0)
    expected_top_k = int(report.get("top_k") or report["probe"].get("experts_per_token") or 0)
    expected_selected_loads = expected_layers * expected_top_k
    baseline_20b = {
        "model": "openai/gpt-oss-20b",
        "measured_decode_seconds": 27.871563799970318,
        "measured_selected_expert_loads": 96,
        "measured_packed_expert_bytes": 1270702080,
    }
    selected_bytes = int(report["selected_expert_bytes_per_token"] or 0)
    baseline_bytes = max(1, baseline_20b["measured_packed_expert_bytes"])
    estimated_seconds = baseline_20b["measured_decode_seconds"] * max(1.0, selected_bytes / baseline_bytes)
    shard_report = _checkpoint_shard_plan(args, download=False)

    return {
        "mode": "preflight",
        "runtime_label": "GPT-OSS low-level runtime smoke test, not an output quality benchmark",
        "model": args.model,
        "runtime_environment": _runtime_environment(args),
        "expected_layers": expected_layers,
        "expected_selected_expert_loads": expected_selected_loads,
        "expected_top_k": expected_top_k,
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "kv_cache_estimate_bytes": kv_estimate,
        "expected_dense_load_estimate_bytes": report["probe"]["estimates"]["estimated_dense_checkpoint_bytes"],
        "expected_selected_expert_load_bytes_per_token": report["selected_expert_bytes_per_token"],
        "expected_dense_fallback_bytes_per_token": int((report["dense_gb_per_token"] or 0) * 1024 ** 3),
        "expected_avoided_bytes_per_token": report["bytes_avoided"],
        "expected_dequant_temporary_bytes": selected_bytes,
        "largest_selected_expert_layer_bytes": largest_dequant_temp,
        "expected_disk_pressure": shard_report,
        "expected_shard_downloads": shard_report["missing_or_invalid_shards"],
        "estimated_seconds_per_token_from_20b_baseline": estimated_seconds,
        "baseline_used": baseline_20b,
        "available_ram_bytes": available_ram,
        "safe_available_ram_bytes": safe_ram,
        "cuda_free_bytes": cuda_free,
        "cuda_total_bytes": cuda_total,
        "windows_memory_guard_triggered": bool(guard_failures),
        "guard_failures": guard_failures,
        "strict_manifest": report["probe"]["manifest_verification"],
        "strict_selective_runtime_can_enable": report["strict_selective_runtime_can_enable"],
        "layout_report": report,
}


def _checkpoint_shard_plan(args, *, download: bool):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for shard planning") from exc

    index = load_safetensors_index(args.model, cache_dir=args.hf_cache_dir, token=args.hf_token)
    if not index:
        raise SystemExit("model.safetensors.index.json is required for shard planning")
    weight_map = index.get("weight_map") or {}
    shard_names = sorted(set(weight_map.values()))
    total_size = int((index.get("metadata") or {}).get("total_size") or 0)
    size_by_name = {}
    local_root = Path(str(args.model)).expanduser() if _is_local_model_dir(args.model) else None
    if local_root is not None:
        for shard in shard_names:
            path = local_root / shard
            if path.exists():
                size_by_name[shard] = int(path.stat().st_size)
    else:
        try:
            info = HfApi().model_info(args.model, files_metadata=True, token=args.hf_token)
            for sibling in info.siblings:
                if sibling.rfilename in shard_names:
                    size = getattr(sibling, "size", None)
                    if size is not None:
                        size_by_name[sibling.rfilename] = int(size)
        except Exception:
            pass

    cache_anchor = Path(args.hf_cache_dir) if args.hf_cache_dir else (REPO_ROOT / ".hf-cache")
    cache_anchor.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(cache_anchor)
    safety_reserve = max(10 * 1024 ** 3, int(total_size * 0.05))
    shards = []
    present_bytes = 0
    missing_expected_bytes = 0
    for shard in shard_names:
        reason_keys = [key for key, mapped in weight_map.items() if mapped == shard][:8]
        path = None
        if local_root is not None:
            path = local_root / shard
        elif download:
            path = Path(hf_hub_download(args.model, shard, cache_dir=args.hf_cache_dir, token=args.hf_token))
        else:
            try:
                path = Path(hf_hub_download(
                    args.model,
                    shard,
                    cache_dir=args.hf_cache_dir,
                    token=args.hf_token,
                    local_files_only=True,
                ))
            except Exception:
                path = None
        size = _path_size(path) if path is not None else None
        expected_size = size_by_name.get(shard) or size
        present = bool(path is not None and path.exists() and size and size > 0)
        if present:
            present_bytes += int(size)
        else:
            missing_expected_bytes += int(expected_size or 0)
        shards.append({
            "filename": shard,
            "present": present,
            "path": str(path) if path is not None else None,
            "size_bytes": size,
            "expected_size_bytes": expected_size,
            "needed_for_sample_keys": reason_keys,
            "resume_can_continue_after_download": True,
        })

    if missing_expected_bytes == 0 and total_size and present_bytes < total_size:
        missing_expected_bytes = max(0, total_size - present_bytes)
    return {
        "total_shards": len(shard_names),
        "index_total_size": total_size,
        "present_bytes": present_bytes,
        "missing_expected_download_bytes": missing_expected_bytes,
        "free_disk_bytes": disk.free,
        "disk_safety_reserve_bytes": safety_reserve,
        "disk_safe_for_missing_downloads": disk.free > missing_expected_bytes + safety_reserve,
        "missing_or_invalid_shards": [item["filename"] for item in shards if not item["present"]],
        "shards": shards,
    }


def _path_size(path: Path):
    try:
        return path.resolve().stat().st_size
    except Exception:
        try:
            return path.stat().st_size
        except Exception:
            return None


def prefetch_required_shards(args):
    if not args.confirm_download:
        raise SystemExit("--prefetch-required-shards requires --confirm-download")
    plan = _checkpoint_shard_plan(args, download=False)
    if not plan["disk_safe_for_missing_downloads"]:
        return {
            "mode": "prefetch_required_shards",
            "model": args.model,
            "status": "clean_abort",
            "reason": "free disk space is below the safety budget for missing checkpoint shards",
            **plan,
        }
    downloaded_plan = _checkpoint_shard_plan(args, download=True)
    report = {
        "mode": "prefetch_required_shards",
        "model": args.model,
        "total_shards": downloaded_plan["total_shards"],
        "index_total_size": downloaded_plan["index_total_size"],
        "total_download_size_bytes": plan["missing_expected_download_bytes"],
        "free_disk_bytes_before": plan["free_disk_bytes"],
        "disk_safety_reserve_bytes": plan["disk_safety_reserve_bytes"],
        "hash_verification": "not_available_in_model.safetensors.index.json",
        "shards": downloaded_plan["shards"],
    }
    for shard in report["shards"]:
        shard["hash_verified"] = False
        shard["hash_reason"] = "the safetensors index does not publish per-shard hashes"
    report["missing_or_invalid_shards"] = downloaded_plan["missing_or_invalid_shards"]
    report["all_required_shards_present"] = not report["missing_or_invalid_shards"]
    return report


def _split_status(args):
    try:
        if _is_local_model_dir(args.model):
            config_path = Path(str(args.model)).expanduser() / "config.json"
        else:
            from huggingface_hub import hf_hub_download
            config_path = Path(hf_hub_download(args.model, "config.json", cache_dir=args.hf_cache_dir, token=args.hf_token))
        split_dir = config_path.parent / "splitted_model.moe"
    except Exception:
        split_dir = None

    manifest_path = split_dir / "moe_expert_index.json" if split_dir is not None else None
    manifest_exists = bool(manifest_path and manifest_path.exists())
    layer_files = list(split_dir.glob("model.layers.*.safetensors")) if split_dir and split_dir.exists() else []
    dense_files = [path for path in layer_files if ".mlp.experts." not in path.name]
    expert_files = list(split_dir.glob("model.layers.*.mlp.experts.*.safetensors")) if split_dir and split_dir.exists() else []
    return {
        "split_dir": str(split_dir) if split_dir else None,
        "manifest_exists": manifest_exists,
        "dense_layer_files": len(dense_files),
        "expert_files": len(expert_files),
        "expected_dense_layer_files": 24,
        "expected_expert_files": 24 * 32,
        "split_ready": bool(manifest_exists and len(dense_files) >= 24 and len(expert_files) >= 24 * 32),
    }


def _assert_preflight_safe(preflight):
    if preflight["guard_failures"]:
        raise MemoryError("; ".join(preflight["guard_failures"]))
    manifest = preflight["strict_manifest"]
    if not preflight["strict_selective_runtime_can_enable"]:
        raise RuntimeError("strict selective runtime cannot be enabled")
    expected = {
        "selective_fused_runtime": True,
        "adapter_name": "gpt_oss_mxfp4_reference",
        "mxfp4_execution": "reference_dequant",
        "dense_contains_full_fused_experts": False,
        "requires_dense_fallback": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"strict manifest assertion failed: {key}={manifest.get(key)!r}, expected {value!r}")


def _harmony_input_ids(model, prompt, *, use_harmony_prompt=False):
    tokenizer = model.tokenizer
    if not use_harmony_prompt:
        token_id = tokenizer.bos_token_id
        if token_id is None:
            token_id = tokenizer.eos_token_id
        if token_id is None:
            token_id = 0
        return torch.tensor([[int(token_id)]], dtype=torch.long), (
            "Using a single raw token for a low-level runtime benchmark, not an output quality benchmark."
        )
    messages = [{"role": "user", "content": prompt}]
    def _as_input_ids(tokenized):
        if hasattr(tokenized, "input_ids"):
            return tokenized.input_ids
        if isinstance(tokenized, dict) and "input_ids" in tokenized:
            return tokenized["input_ids"]
        return tokenized

    if getattr(tokenizer, "chat_template", None):
        try:
            tokenized = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            return _as_input_ids(tokenized), None
        except Exception as exc:
            return tokenizer(prompt, return_tensors="pt").input_ids, (
                f"Harmony/chat template failed ({exc}). This is a low-level runtime benchmark, not an output quality benchmark."
            )
    return tokenizer(prompt, return_tensors="pt").input_ids, (
        "This is a low-level runtime benchmark, not an output quality benchmark."
    )


def _decode_generated_text(model, input_ids, output):
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None, None
    try:
        input_token_count = int(input_ids.shape[-1])
        output_cpu = output.detach().cpu()
        generated_ids = output_cpu[0, input_token_count:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        decoded_text = tokenizer.decode(output_cpu[0], skip_special_tokens=True)
        return generated_text, decoded_text
    except Exception as exc:
        return None, f"<decode failed: {type(exc).__name__}: {exc}>"


def _assert_runtime_strict(model, args):
    stats = model.runtime_stats()
    assertions = {
        "selective_fused_runtime": True,
        "adapter_name": "gpt_oss_mxfp4_reference",
        "mxfp4_execution": "reference_dequant",
        "dense_contains_full_fused_experts": False,
        "requires_dense_fallback": False,
        "strict_full_fused_tensor_loaded": False,
    }
    for key, value in assertions.items():
        if stats.get(key) != value:
            raise RuntimeError(f"runtime strict assertion failed: {key}={stats.get(key)!r}, expected {value!r}")
    if args.moe_allow_dense_fallback:
        raise RuntimeError("moe_allow_dense_fallback must be false for strict one-token smoke")
    return stats


def run_one_layer_smoke(args, preflight):
    _assert_preflight_safe(preflight)
    started = time.perf_counter()
    result = download_gpt_oss_one_expert_mxfp4_smoke(
        args.model,
        layer_id=args.layer_id,
        expert_id=args.expert_id,
        cache_dir=args.hf_cache_dir,
        token=args.hf_token,
    )
    result["mode"] = "one_layer_smoke"
    result["seconds"] = time.perf_counter() - started
    result["preflight"] = preflight
    result["strict_full_fused_tensor_loaded"] = False
    result["dense_fallback_used"] = False
    result["expert_materialization_mode"] = args.expert_materialization_mode
    return result


def run_one_token_real(args, preflight):
    _assert_preflight_safe(preflight)
    if not args.confirm_real_run:
        raise SystemExit("--one-token-real requires --confirm-real-run")
    max_real_new_tokens = 256 if args.use_kv_cache else 32
    if args.max_new_tokens < 1 or args.max_new_tokens > max_real_new_tokens:
        raise SystemExit(
            f"This smoke script permits --max-new-tokens from 1 to {max_real_new_tokens} for real runs"
        )
    if args.warm_cache_second_token_test and args.max_new_tokens != 1:
        raise SystemExit("--warm-cache-second-token-test is only supported with --max-new-tokens 1")
    split_status = _split_status(args)
    if not split_status["split_ready"] and not args.allow_split_build and not args.lazy_build:
        return {
            "mode": "one_token_real",
            "status": "clean_abort",
            "use_kv_cache": bool(args.use_kv_cache),
            "reason": "MoE split is not complete; refusing a long first-run split build without --allow-split-build or --lazy-build.",
            "suggestion": "Run --one-layer-smoke first, rerun with --lazy-build, or use --allow-split-build when you are ready for a long disk build.",
            "split_status": split_status,
            "preflight": preflight,
            "peak_vram_mb": _cuda_peak_mb(),
        }

    kwargs = {
        "device": args.dense_device or args.device,
        "dtype": _dtype_from_name(args.dtype),
        "max_seq_len": args.max_seq_len,
        "prefetching": not args.no_prefetching,
        "moe_strict_streaming": args.moe_strict_streaming,
        "moe_allow_dense_fallback": args.moe_allow_dense_fallback,
        "moe_expert_cache_mb": args.moe_expert_cache_mb,
        "moe_cpu_expert_cache_mb": args.moe_cpu_expert_cache_mb,
        "os_reserved_ram_mb": args.os_reserved_ram_mb,
        "abort_unsafe_context": True,
        "resume_split_build": args.resume_split_build,
        "lazy_build": args.lazy_build,
        "expert_materialization_mode": args.expert_materialization_mode,
        "layer_cleanup_interval": args.layer_cleanup_interval,
        "mxfp4_device": args.mxfp4_device,
        "expert_matmul_device": args.expert_matmul_device,
        "dense_device": args.dense_device,
        "max_vram_mb": args.max_vram_mb,
        "sync_cuda_timing": args.sync_cuda_timing,
        "enable_gpt_oss_batch_direct_slice": args.enable_gpt_oss_batch_direct_slice,
        "keep_nontransformer_resident": args.keep_nontransformer_resident,
    }
    if args.layer_shards_saving_path:
        kwargs["layer_shards_saving_path"] = args.layer_shards_saving_path
    if args.hf_token:
        kwargs["hf_token"] = args.hf_token

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with MemorySampler() as sampler:
            model = AutoModel.from_pretrained(args.model, **kwargs)
            strict_stats_before = _assert_runtime_strict(model, args)
            input_ids, harmony_warning = _harmony_input_ids(
                model,
                args.prompt,
                use_harmony_prompt=args.use_harmony_prompt,
            )
            input_ids = input_ids.to(model.device)
            input_token_count = int(input_ids.shape[-1])
            if input_token_count > args.max_seq_len:
                raise MemoryError(f"prompt tokens {input_token_count} exceed max_seq_len {args.max_seq_len}")
            if input_token_count + int(args.max_new_tokens) > args.max_seq_len:
                raise MemoryError(
                    f"prompt tokens {input_token_count} + max_new_tokens {args.max_new_tokens} exceed max_seq_len {args.max_seq_len}"
                )
            model.clear_runtime_caches()
            warm_cache_result = None
            if args.warm_selected_expert_cache and hasattr(model, "warm_selected_expert_cache"):
                warm_cache_result = model.warm_selected_expert_cache()
            decode_started = time.perf_counter()
            attention_mask = torch.ones_like(input_ids, device=model.device)
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                use_cache=bool(args.use_kv_cache),
            )
            if torch.cuda.is_available() and str(model.device).startswith("cuda"):
                torch.cuda.synchronize(model.device)
            decode_elapsed = time.perf_counter() - decode_started
            if hasattr(model, "flush_runtime_progress"):
                model.flush_runtime_progress()
            stats = _assert_runtime_strict(model, args)
            output_token_count = int(output.shape[-1])
            generated_tokens = max(0, output_token_count - input_token_count)
            first_pass_summary = _stats_summary(stats, decode_elapsed, generated_tokens)
            second_pass_summary = None
            total_two_pass_decode_seconds = None
            if args.warm_cache_second_token_test:
                first_decode_elapsed = decode_elapsed
                model.reset_runtime_stats()
                second_input = output.to(model.device)
                second_input_token_count = int(second_input.shape[-1])
                second_decode_started = time.perf_counter()
                second_output = model.generate(
                    second_input,
                    attention_mask=torch.ones_like(second_input, device=model.device),
                    max_new_tokens=1,
                    use_cache=bool(args.use_kv_cache),
                )
                if torch.cuda.is_available() and str(model.device).startswith("cuda"):
                    torch.cuda.synchronize(model.device)
                second_decode_elapsed = time.perf_counter() - second_decode_started
                if hasattr(model, "flush_runtime_progress"):
                    model.flush_runtime_progress()
                second_stats = _assert_runtime_strict(model, args)
                second_generated_tokens = max(0, int(second_output.shape[-1]) - second_input_token_count)
                second_pass_summary = _stats_summary(second_stats, second_decode_elapsed, second_generated_tokens)
                total_two_pass_decode_seconds = first_decode_elapsed + second_decode_elapsed
                output = second_output
                stats = second_stats
                decode_elapsed = second_decode_elapsed
                generated_tokens = second_generated_tokens
            generated_text, decoded_text = _decode_generated_text(model, input_ids, output)
            num_layers = int(preflight["layout_report"]["probe"].get("num_hidden_layers") or 0)
            num_experts = int(preflight["layout_report"]["probe"].get("num_experts") or 32)
            top_k = int(preflight["layout_report"].get("top_k") or 4)
            selected_builds = int(stats.get("selected_expert_builds_this_run", 0))
            unused_builds = int(stats.get("unused_expert_builds_this_run", 0))
            if args.use_kv_cache:
                generation_work_tokens = input_token_count + max(0, int(args.max_new_tokens) - 1)
            else:
                generation_work_tokens = sum(
                    range(max(1, input_token_count), max(1, input_token_count) + int(args.max_new_tokens))
                )
            materialization_limit = max(
                num_layers * top_k * max(1, generation_work_tokens) * args.expert_materialization_safety_factor,
                num_layers * num_experts * args.expert_materialization_safety_factor,
            )
            if unused_builds != 0 or selected_builds > materialization_limit:
                raise RuntimeError(
                    "Lazy path is building too many experts. Routing-first selected expert materialization is not working. "
                    f"selected_expert_builds_this_run={selected_builds}, unused_expert_builds_this_run={unused_builds}, "
                    f"limit={materialization_limit}"
                )
    except (MemoryError, torch.OutOfMemoryError) as exc:
        return {
            "mode": "one_token_real",
            "status": "clean_abort",
            "use_kv_cache": bool(args.use_kv_cache),
            "reason": str(exc),
            "suggestion": "Lower --moe-expert-cache-mb, --moe-cpu-expert-cache-mb, or --max-seq-len.",
            "preflight": preflight,
            "peak_vram_mb": _cuda_peak_mb(),
        }
    except Exception as exc:
        missing_hint = ""
        if "Missing original checkpoint shard:" in str(exc):
            missing_hint = str(exc)
        return {
            "mode": "one_token_real",
            "status": "clean_abort",
            "use_kv_cache": bool(args.use_kv_cache),
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "suggestion": missing_hint or "Check the reported layer/expert path and strict manifest fields before retrying.",
            "preflight": preflight,
            "peak_vram_mb": _cuda_peak_mb(),
        }

    total_elapsed = time.perf_counter() - started
    denom = max(1, generated_tokens)
    layer_timing_breakdown = _merge_layer_timing_breakdown(stats)
    timing_accounting = _timing_accounting(stats, decode_elapsed, layer_timing_breakdown)
    device_report = _device_report(args, stats)
    return {
        "mode": "one_token_real",
        "status": "runtime_completed",
        "runtime_label": "GPT-OSS low-level runtime smoke test, not an output quality benchmark",
        "harmony_warning": harmony_warning,
        "prompt_tokens": input_token_count,
        "requested_max_new_tokens": args.max_new_tokens,
        "use_kv_cache": bool(args.use_kv_cache),
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "decoded_text": decoded_text,
        "total_seconds": total_elapsed,
        "decode_seconds": decode_elapsed,
        "seconds_per_token": decode_elapsed / denom,
        "tokens_per_second": generated_tokens / decode_elapsed if decode_elapsed > 0 else 0.0,
        "cpu_bytes_loaded": stats.get("cpu_load_bytes", 0),
        "device_bytes_loaded": stats.get("device_load_bytes", 0),
        "packed_expert_bytes_loaded": stats.get("packed_expert_bytes_loaded", 0),
        "dequantized_temporary_bytes": stats.get("dequantized_temporary_bytes", 0),
        "mxfp4_dequant_seconds": stats.get("mxfp4_dequant_seconds", 0.0),
        "expert_matmul_seconds": stats.get("expert_matmul_seconds", 0.0),
        "gate_up_matmul_seconds": stats.get("gate_up_matmul_seconds", 0.0),
        "down_matmul_seconds": stats.get("down_matmul_seconds", 0.0),
        "activation_seconds": stats.get("activation_seconds", 0.0),
        "top_bottlenecks": {
            "total_direct_slice_read_time": stats.get("selected_expert_materialization_time_this_run", 0.0),
            "total_dequant_time": stats.get("mxfp4_dequant_seconds", 0.0),
            "total_matmul_time": stats.get("expert_matmul_seconds", 0.0),
            "total_attention_time": timing_accounting["attention_seconds"],
            "total_router_time": timing_accounting["router_seconds"],
            "total_cleanup_time": timing_accounting["cleanup_seconds"],
            "total_streaming_wait_time": timing_accounting["streaming_wait_seconds"],
            "non_layer_generate_overhead_time": timing_accounting["non_layer_generate_overhead_seconds"],
            "total_checkpoint_wait_download_time": None,
            "slowest_layer": _slowest_layer(layer_timing_breakdown),
        },
        "timing_accounting": timing_accounting,
        "layer_timing_breakdown": layer_timing_breakdown,
        "gpu_execution_check": device_report,
        "dense_bytes_avoided": preflight["expected_avoided_bytes_per_token"],
        "selected_experts_per_layer": stats.get("fused_adapter_stats", {}).get("selected_expert_slots", 0),
        "total_selected_experts_requested": stats.get("fused_adapter_stats", {}).get("selected_expert_slots", 0),
        "total_unique_experts_materialized": stats.get("selected_expert_builds_this_run", 0),
        "total_unused_experts_materialized": stats.get("unused_expert_builds_this_run", 0),
        "expert_materialization_mode": stats.get("expert_materialization_mode"),
        "direct_slice_expert_loads_this_run": stats.get("direct_slice_expert_loads_this_run", 0),
        "direct_slice_packed_bytes_read_this_run": stats.get("direct_slice_packed_bytes_read_this_run", 0),
        "selected_expert_materialization_time_this_run": stats.get("selected_expert_materialization_time_this_run", 0.0),
        "cache_hits": stats.get("expert_cache_hits", 0),
        "cache_misses": stats.get("expert_cache_misses", 0),
        "cpu_cache_hits": stats.get("cpu_expert_cache_hits", 0),
        "cpu_cache_misses": stats.get("cpu_expert_cache_misses", 0),
        "peak_ram_mb": (sampler.peak_rss / 1024 ** 2) if sampler.peak_rss is not None else None,
        "peak_tracemalloc_mb": sampler.peak_tracemalloc_bytes / 1024 ** 2,
        "peak_vram_mb": _cuda_peak_mb(),
        "peak_cuda_allocated_mb": stats.get("peak_cuda_allocated_bytes", 0) / 1024 ** 2,
        "peak_cuda_reserved_mb": stats.get("peak_cuda_reserved_bytes", 0) / 1024 ** 2,
        "max_vram_mb": args.max_vram_mb,
        "vram_budget_respected": (
            True if args.max_vram_mb is None else stats.get("peak_cuda_reserved_bytes", 0) <= args.max_vram_mb * 1024 ** 2
        ),
        "kv_cache_estimate_bytes": preflight["kv_cache_estimate_bytes"],
        "windows_memory_guard_triggered": preflight["windows_memory_guard_triggered"],
        "strict_mode": True,
        "layer_cleanup_interval": args.layer_cleanup_interval,
        "dense_fallback_used": False,
        "strict_stats_before": strict_stats_before,
        "runtime_stats": stats,
        "warm_selected_expert_cache": warm_cache_result,
        "warm_cache_second_token_test": {
            "enabled": bool(args.warm_cache_second_token_test),
            "first_pass": first_pass_summary,
            "second_pass": second_pass_summary,
            "total_two_pass_decode_seconds": total_two_pass_decode_seconds,
            "second_pass_faster": (
                bool(second_pass_summary and second_pass_summary["seconds_per_token"] < first_pass_summary["seconds_per_token"])
            ),
        },
        "preflight": preflight,
    }


def _merge_layer_timing_breakdown(stats):
    adapter_layers = stats.get("fused_adapter_stats", {}).get("layer_timing_breakdown", {}) or {}
    base_layers = stats.get("layer_runtime_breakdown", {}) or {}
    timed_modules = stats.get("timed_module_stats", {}) or {}
    split_layers = stats.get("split_builder_status", {})
    progress_path = split_layers.get("progress_path")
    persisted_timings = {}
    if progress_path and Path(progress_path).exists():
        try:
            persisted_timings = json.loads(Path(progress_path).read_text(encoding="utf-8")).get("layer_timings", {})
        except Exception:
            persisted_timings = {}
    merged = {}
    all_layer_names = sorted(set(adapter_layers) | set(base_layers) | set(timed_modules))
    for layer_name in all_layer_names:
        layer_stats = adapter_layers.get(layer_name, {})
        base_stats = base_layers.get(layer_name, {})
        module_stats = timed_modules.get(layer_name, {})
        layer_progress = persisted_timings.get(layer_name + "." if not layer_name.endswith(".") else layer_name, {})
        selected = layer_progress.get("selected_experts", {}) if isinstance(layer_progress, dict) else {}
        direct_slice_read_time = sum(float(item.get("seconds", 0.0)) for item in selected.values() if isinstance(item, dict))
        packed_bytes_read = sum(int(item.get("packed_bytes_read", 0)) for item in selected.values() if isinstance(item, dict))
        dense_ready = layer_progress.get("dense_ready", {}) if isinstance(layer_progress, dict) else {}
        layer_forward_time = float(base_stats.get("layer_forward_seconds", 0.0))
        attention_time = float(module_stats.get("attention_seconds", 0.0))
        router_time = float(module_stats.get("router_seconds", 0.0))
        mlp_total_time = float(module_stats.get("mlp_total_seconds", 0.0))
        dequant_time = float(layer_stats.get("mxfp4_dequant_seconds", 0.0))
        matmul_time = float(layer_stats.get("expert_matmul_seconds", 0.0))
        activation_time = float(layer_stats.get("activation_seconds", 0.0))
        input_cast_time = float(layer_stats.get("input_cast_seconds", 0.0))
        expert_known_time = direct_slice_read_time + dequant_time + matmul_time + activation_time + input_cast_time
        merged[layer_name] = {
            "selected_experts": layer_stats.get("selected_experts", []),
            "selected_expert_count": layer_stats.get("selected_expert_count", 0),
            "dense_load_time": base_stats.get("dense_load_seconds", dense_ready.get("seconds")),
            "prefetch_wait_time": base_stats.get("prefetch_wait_seconds"),
            "move_to_device_time": base_stats.get("move_to_device_seconds"),
            "position_args_time": base_stats.get("position_args_seconds"),
            "layer_forward_time": layer_forward_time,
            "attention_time": attention_time,
            "router_time": router_time,
            "mlp_total_time": mlp_total_time,
            "direct_slice_read_time": direct_slice_read_time,
            "packed_bytes_read": packed_bytes_read,
            "mxfp4_dequant_time": dequant_time,
            "input_cast_time": input_cast_time,
            "expert_matmul_time": matmul_time,
            "gate_up_matmul_time": layer_stats.get("gate_up_matmul_seconds", 0.0),
            "down_matmul_time": layer_stats.get("down_matmul_seconds", 0.0),
            "activation_time": activation_time,
            "offload_time": base_stats.get("offload_seconds"),
            "cleanup_time": base_stats.get("cleanup_seconds"),
            "layer_loop_time": base_stats.get("layer_loop_seconds"),
            "expert_python_overhead_time": max(0.0, mlp_total_time - expert_known_time) if mlp_total_time else None,
            "non_attention_mlp_forward_time": max(0.0, layer_forward_time - attention_time - mlp_total_time),
            "total_layer_time_known_parts": direct_slice_read_time + dequant_time + matmul_time + activation_time + input_cast_time,
            "attention_input_device": module_stats.get("attention_seconds_input_device"),
            "attention_output_device": module_stats.get("attention_seconds_output_device"),
            "router_input_device": module_stats.get("router_seconds_input_device"),
            "router_output_device": module_stats.get("router_seconds_output_device"),
        }
    return merged


def _slowest_layer(layer_timing_breakdown):
    if not layer_timing_breakdown:
        return None
    layer_name, layer_stats = max(
        layer_timing_breakdown.items(),
        key=lambda item: item[1].get("total_layer_time_known_parts", 0.0),
    )
    return {"layer": layer_name, **layer_stats}


def _stats_summary(stats, decode_elapsed, generated_tokens):
    denom = max(1, int(generated_tokens))
    layer_timing_breakdown = _merge_layer_timing_breakdown(stats)
    timing_accounting = _timing_accounting(stats, decode_elapsed, layer_timing_breakdown)
    return {
        "generated_tokens": int(generated_tokens),
        "decode_seconds": decode_elapsed,
        "seconds_per_token": decode_elapsed / denom,
        "tokens_per_second": int(generated_tokens) / decode_elapsed if decode_elapsed > 0 else 0.0,
        "direct_slice_expert_loads_this_run": stats.get("direct_slice_expert_loads_this_run", 0),
        "packed_expert_bytes_loaded": stats.get("packed_expert_bytes_loaded", 0),
        "dequantized_temporary_bytes": stats.get("dequantized_temporary_bytes", 0),
        "mxfp4_dequant_seconds": stats.get("mxfp4_dequant_seconds", 0.0),
        "expert_matmul_seconds": stats.get("expert_matmul_seconds", 0.0),
        "cache_hits": stats.get("expert_cache_hits", 0),
        "cache_misses": stats.get("expert_cache_misses", 0),
        "cpu_cache_hits": stats.get("cpu_expert_cache_hits", 0),
        "cpu_cache_misses": stats.get("cpu_expert_cache_misses", 0),
        "timing_accounting": timing_accounting,
        "slowest_layer": _slowest_layer(layer_timing_breakdown),
    }


def _timing_accounting(stats, decode_elapsed, layer_timing_breakdown):
    adapter = stats.get("fused_adapter_stats", {}) or {}
    layer_loop = float(stats.get("layer_loop_seconds", 0.0))
    layer_forward = float(stats.get("layer_forward_seconds", 0.0))
    attention = sum(float(item.get("attention_time") or 0.0) for item in layer_timing_breakdown.values())
    router = sum(float(item.get("router_time") or 0.0) for item in layer_timing_breakdown.values())
    mlp_total = sum(float(item.get("mlp_total_time") or 0.0) for item in layer_timing_breakdown.values())
    direct_slice = float(stats.get("selected_expert_materialization_time_this_run", 0.0))
    dequant = float(stats.get("mxfp4_dequant_seconds", 0.0))
    matmul = float(stats.get("expert_matmul_seconds", 0.0))
    activation = float(stats.get("activation_seconds", 0.0))
    input_cast = float(adapter.get("input_cast_seconds", 0.0))
    expert_known = direct_slice + dequant + matmul + activation + input_cast
    return {
        "decode_seconds": decode_elapsed,
        "layer_loop_seconds": layer_loop,
        "non_layer_generate_overhead_seconds": decode_elapsed - layer_loop,
        "overlapped_background_cpu_load_seconds": stats.get("cpu_load_seconds", 0.0),
        "streaming_wait_seconds": stats.get("prefetch_wait_seconds", 0.0),
        "move_to_device_seconds": stats.get("device_load_seconds", 0.0),
        "position_args_seconds": stats.get("position_args_seconds", 0.0),
        "offload_seconds": stats.get("offload_seconds", 0.0),
        "cleanup_seconds": stats.get("cleanup_seconds", 0.0),
        "layer_forward_seconds": layer_forward,
        "attention_seconds": attention,
        "router_seconds": router,
        "mlp_total_seconds": mlp_total,
        "direct_slice_read_seconds": direct_slice,
        "mxfp4_dequant_seconds": dequant,
        "expert_matmul_seconds": matmul,
        "activation_seconds": activation,
        "input_cast_seconds": input_cast,
        "expert_python_overhead_seconds": max(0.0, mlp_total - expert_known) if mlp_total else None,
        "non_attention_mlp_forward_seconds": max(0.0, layer_forward - attention - mlp_total),
    }


def _device_report(args, stats):
    adapter_report = (stats.get("fused_adapter_stats", {}) or {}).get("device_report", {}) or {}
    timed_modules = stats.get("timed_module_stats", {}) or {}
    attention_devices = sorted({
        value
        for layer in timed_modules.values()
        for key, value in layer.items()
        if key.startswith("attention_seconds_") and key.endswith("_device") and value is not None
    })
    router_devices = sorted({
        value
        for layer in timed_modules.values()
        for key, value in layer.items()
        if key.startswith("router_seconds_") and key.endswith("_device") and value is not None
    })
    return {
        "torch_cuda_available": torch.cuda.is_available(),
        "requested_device": args.device,
        "hidden_states_device": adapter_report.get("hidden_states_device"),
        "dequantized_expert_weight_devices": {
            "gate_up": adapter_report.get("dequantized_gate_up_device"),
            "down": adapter_report.get("dequantized_down_device"),
        },
        "matmul_device": adapter_report.get("matmul_device"),
        "packed_expert_devices": {
            "gate_up_blocks": adapter_report.get("packed_gate_up_blocks_device"),
            "down_blocks": adapter_report.get("packed_down_blocks_device"),
        },
        "attention_tensor_devices": attention_devices,
        "router_tensor_devices": router_devices,
        "compute_dtype": adapter_report.get("compute_dtype"),
        "mxfp4_execution": stats.get("mxfp4_execution"),
        "cuda_memory_allocated_bytes": adapter_report.get("cuda_memory_allocated_bytes"),
        "cuda_memory_reserved_bytes": adapter_report.get("cuda_memory_reserved_bytes"),
        "cuda_max_memory_allocated_bytes": adapter_report.get("cuda_max_memory_allocated_bytes"),
        "cuda_max_memory_reserved_bytes": adapter_report.get("cuda_max_memory_reserved_bytes"),
    }


def compare_cpu_cuda(args):
    if not args.confirm_real_run:
        raise SystemExit("--compare-cpu-cuda requires --confirm-real-run")
    if not torch.cuda.is_available():
        raise SystemExit("--compare-cpu-cuda requires torch.cuda.is_available() == true")

    cpu_args = copy.copy(args)
    cpu_args.compare_cpu_cuda = False
    cpu_args.device = "cpu"
    cpu_args.dense_device = "cpu"
    cpu_args.mxfp4_device = "cpu"
    cpu_args.expert_matmul_device = "cpu"
    cpu_args.max_vram_mb = None

    cuda_args = copy.copy(args)
    cuda_args.compare_cpu_cuda = False
    cuda_args.device = args.device if str(args.device).startswith("cuda") else "cuda:0"
    cuda_args.dense_device = args.dense_device or cuda_args.device
    cuda_args.mxfp4_device = args.mxfp4_device or cuda_args.device
    cuda_args.expert_matmul_device = args.expert_matmul_device or cuda_args.mxfp4_device

    cpu_preflight = _build_preflight(cpu_args)
    cpu_result = run_one_token_real(cpu_args, cpu_preflight)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    cuda_preflight = _build_preflight(cuda_args)
    cuda_result = run_one_token_real(cuda_args, cuda_preflight)
    return {
        "mode": "compare_cpu_cuda",
        "model": args.model,
        "runtime_label": "GPT-OSS low-level runtime benchmark, not an output quality benchmark",
        "cpu": _comparison_summary(cpu_result),
        "cuda": _comparison_summary(cuda_result),
        "cpu_result": cpu_result,
        "cuda_result": cuda_result,
    }


def _comparison_summary(result):
    stats = result.get("timing_accounting", {}) if isinstance(result, dict) else {}
    return {
        "status": result.get("status"),
        "seconds_per_token": result.get("seconds_per_token"),
        "mxfp4_dequant_seconds": result.get("mxfp4_dequant_seconds"),
        "expert_matmul_seconds": result.get("expert_matmul_seconds"),
        "direct_slice_read_seconds": stats.get("direct_slice_read_seconds"),
        "python_overhead_seconds": stats.get("expert_python_overhead_seconds"),
        "peak_ram_mb": result.get("peak_ram_mb"),
        "peak_vram_mb": result.get("peak_vram_mb"),
        "peak_cuda_allocated_mb": result.get("peak_cuda_allocated_mb"),
        "peak_cuda_reserved_mb": result.get("peak_cuda_reserved_mb"),
        "dense_fallback_used": result.get("dense_fallback_used"),
        "strict_full_fused_tensor_loaded": (
            result.get("runtime_stats", {}).get("strict_full_fused_tensor_loaded")
            if isinstance(result.get("runtime_stats"), dict)
            else None
        ),
        "unused_experts": result.get("total_unused_experts_materialized"),
    }


def _print_real_run_result(args, result):
    if not args.print_generated_text:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    summary = {
        "status": result.get("status"),
        "generated_tokens": result.get("generated_tokens"),
        "use_kv_cache": result.get("use_kv_cache"),
        "decode_seconds": result.get("decode_seconds"),
        "seconds_per_token": result.get("seconds_per_token"),
        "tokens_per_second": result.get("tokens_per_second"),
        "peak_vram_mb": result.get("peak_vram_mb"),
        "dense_fallback_used": result.get("dense_fallback_used"),
        "strict_full_fused_tensor_loaded": (
            result.get("runtime_stats", {}).get("strict_full_fused_tensor_loaded")
            if isinstance(result.get("runtime_stats"), dict)
            else None
        ),
        "reason": result.get("reason"),
    }
    print(json.dumps({"run_summary": summary}, indent=2, sort_keys=True), file=sys.stderr)
    if result.get("status") == "runtime_completed":
        print((result.get("generated_text") or "").strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--prompt", default="Hello.")
    parser.add_argument("--use-harmony-prompt", action="store_true", help="Use tokenizer chat/Harmony formatting instead of a single raw runtime token")
    parser.add_argument("--use-kv-cache", action="store_true", help="Use Transformers DynamicCache during generation")
    parser.add_argument("--print-generated-text", action="store_true", help="Print only generated text to stdout; emit compact run summary to stderr")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dense-device", help="Device for streamed dense layers; defaults to --device")
    parser.add_argument("--mxfp4-device", help="Device for MXFP4 unpack/dequant; defaults to dense/device")
    parser.add_argument("--expert-matmul-device", help="Device for selected expert matmuls; defaults to --mxfp4-device")
    parser.add_argument("--max-vram-mb", type=int, help="Abort if CUDA reserved/allocated memory exceeds this budget")
    parser.add_argument("--sync-cuda-timing", action="store_true", help="Synchronize around timed CUDA sub-stages for detailed timings; slower")
    parser.add_argument("--enable-gpt-oss-batch-direct-slice", action="store_true", help="Experimental: load selected GPT-OSS direct-slice experts in layer batches")
    parser.add_argument("--no-keep-nontransformer-resident", dest="keep_nontransformer_resident", action="store_false", help="Do not keep embed/norm/lm_head resident on the dense device during the smoke run")
    parser.set_defaults(keep_nontransformer_resident=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--moe-strict-streaming", action="store_true")
    parser.add_argument("--moe-allow-dense-fallback", action="store_true")
    parser.add_argument("--moe-expert-cache-mb", type=int, default=1024)
    parser.add_argument("--moe-cpu-expert-cache-mb", type=int, default=4096)
    parser.add_argument("--os-reserved-ram-mb", type=int, default=8192)
    parser.add_argument("--hf-token")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--layer-shards-saving-path")
    parser.add_argument("--no-prefetching", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--one-layer-smoke", action="store_true")
    parser.add_argument("--one-token-real", action="store_true")
    parser.add_argument("--compare-cpu-cuda", action="store_true")
    parser.add_argument("--prefetch-required-shards", action="store_true")
    parser.add_argument("--allow-split-build", action="store_true", help="Permit long first-run creation of splitted_model.moe")
    parser.add_argument("--resume-split-build", action="store_true", help="Resume an interrupted incremental split build")
    parser.add_argument("--lazy-build", action="store_true", help="Build missing split shards layer-by-layer during the smoke run")
    parser.add_argument("--expert-materialization-mode", choices=("persisted_split", "direct_slice", "hybrid"), default="direct_slice")
    parser.add_argument("--expert-materialization-safety-factor", type=int, default=2)
    parser.add_argument("--warm-selected-expert-cache", action="store_true", help="Warm only selected experts observed in prior runs")
    parser.add_argument("--warm-cache-second-token-test", action="store_true", help="Generate a second one-token pass without clearing expert caches")
    parser.add_argument("--layer-cleanup-interval", type=int, default=1, help="Run aggressive clean_memory every N streamed layers; 0 disables per-layer cleanup")
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--confirm-real-run", action="store_true")
    parser.add_argument("--confirm-download", action="store_true")
    parser.add_argument("--confirm-120b", action="store_true")
    args = parser.parse_args()

    if args.hf_cache_dir:
        cache_dir = str(Path(args.hf_cache_dir).resolve())
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache_dir) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HUGGINGFACE_HUB_CACHE"])
    _validate_device_args(args)
    startup_env = _runtime_environment(args)
    print(json.dumps({"startup_runtime_environment": startup_env}, sort_keys=True), file=sys.stderr)

    try:
        model_config_for_gate = load_config(args.model, cache_dir=args.hf_cache_dir, token=args.hf_token)
    except Exception as exc:
        raise SystemExit(f"Could not read GPT-OSS config.json from {args.model}: {exc}") from exc
    if not _is_gpt_oss_config(model_config_for_gate):
        raise SystemExit(
            "This smoke script only supports GPT-OSS models. "
            f"config model_type={model_config_for_gate.get('model_type')!r}, "
            f"architectures={model_config_for_gate.get('architectures')!r}"
        )

    if _is_probably_120b(args, model_config_for_gate):
        if not args.confirm_120b:
            raise SystemExit("GPT-OSS 120B smoke paths require --confirm-120b.")
        if args.preflight_only and not args.confirm_real_run:
            raise SystemExit("GPT-OSS 120B preflight requires --confirm-real-run --confirm-120b.")
        if args.one_layer_smoke and not args.confirm_real_run:
            raise SystemExit("GPT-OSS 120B one-layer smoke requires --confirm-real-run --confirm-120b.")
        max_120b_seq_len = 256 if args.use_kv_cache else 128
        if (args.one_token_real or args.compare_cpu_cuda) and (not args.confirm_real_run or args.max_seq_len > max_120b_seq_len):
            raise SystemExit(
                f"GPT-OSS 120B real runs require --confirm-real-run --confirm-120b and --max-seq-len <= {max_120b_seq_len}."
            )
        if (args.one_token_real or args.compare_cpu_cuda) and args.max_new_tokens != 1 and not args.use_kv_cache:
            raise SystemExit("GPT-OSS 120B multi-token real runs require --use-kv-cache; otherwise use --max-new-tokens 1.")
    if (args.one_token_real or args.compare_cpu_cuda) and not args.moe_strict_streaming:
        raise SystemExit("--moe-strict-streaming is required for one-token real runs")

    if args.prefetch_required_shards:
        print(json.dumps(prefetch_required_shards(args), indent=2, sort_keys=True))
        return

    if args.compare_cpu_cuda:
        print(json.dumps(compare_cpu_cuda(args), indent=2, sort_keys=True))
        return

    preflight = _build_preflight(args)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    if args.one_layer_smoke:
        if not args.confirm_real_run:
            raise SystemExit("--one-layer-smoke requires --confirm-real-run because it may download checkpoint shards")
        print(json.dumps(run_one_layer_smoke(args, preflight), indent=2, sort_keys=True))
        return

    if not args.one_token_real:
        args.one_token_real = True
    if not args.confirm_real_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        raise SystemExit("--one-token-real requires --confirm-real-run")
    _print_real_run_result(args, run_one_token_real(args, preflight))


if __name__ == "__main__":
    main()
