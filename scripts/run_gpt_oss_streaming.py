"""Usable GPT-OSS strict direct-slice runner.

This keeps the BetterAirLLM runtime path conservative: no dense fallback, no
full fused expert load, and no core runtime changes.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import queue
import re
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AIR_LLM_ROOT = REPO_ROOT / "air_llm"
if str(AIR_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(AIR_LLM_ROOT))

import torch  # noqa: E402
from airllm import AutoModel  # noqa: E402
from smoke_gpt_oss_20b_one_token import (  # noqa: E402
    _dtype_from_name,
    _is_gpt_oss_config,
    _is_probably_120b,
    _merge_layer_timing_breakdown,
    _runtime_environment,
    _validate_device_args,
    load_config,
)

try:  # noqa: E402
    from transformers import TextIteratorStreamer
except Exception:  # pragma: no cover - old transformers layout fallback
    from transformers.generation.streamers import TextIteratorStreamer  # type: ignore


HARMONY_FINAL_MARKERS = (
    "<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
)


class TimingTextIteratorStreamer(TextIteratorStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_events = []

    def put(self, value):
        prompt_event = self.skip_prompt and self.next_tokens_are_prompt
        if not prompt_event:
            try:
                token_count = int(value.numel())
            except Exception:
                token_count = 1
            self.token_events.append({"timestamp": time.perf_counter(), "token_count": token_count})
        return super().put(value)


def _as_input_ids(tokenized):
    if hasattr(tokenized, "input_ids"):
        return tokenized.input_ids
    if isinstance(tokenized, dict) and "input_ids" in tokenized:
        return tokenized["input_ids"]
    return tokenized


def _build_input_ids(model, prompt, *, use_harmony_prompt, reasoning_effort):
    tokenizer = model.tokenizer
    if use_harmony_prompt and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            tokenized = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                reasoning_effort=reasoning_effort,
            )
            return _as_input_ids(tokenized), None
        except TypeError:
            tokenized = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            return _as_input_ids(tokenized), None
        except Exception as exc:
            warning = f"Harmony prompt formatting failed; falling back to raw prompt: {type(exc).__name__}: {exc}"
            return tokenizer(prompt, return_tensors="pt").input_ids, warning

    return tokenizer(prompt, return_tensors="pt").input_ids, None


def _decode_new_tokens(model, input_ids, output):
    tokenizer = model.tokenizer
    input_token_count = int(input_ids.shape[-1])
    output_cpu = output.detach().cpu()
    generated_ids = output_cpu[0, input_token_count:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True), int(generated_ids.numel())


def _remove_prompt_echo(text, prompt):
    cleaned = text.lstrip()
    prompt_clean = prompt.strip()
    if not prompt_clean:
        return cleaned

    while cleaned.lower().startswith(prompt_clean.lower()):
        cleaned = cleaned[len(prompt_clean) :].lstrip(" \t\r\n:-")
    return cleaned


def _clean_generated_text(text, prompt):
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")

    for marker in HARMONY_FINAL_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.rsplit(marker, 1)[-1]

    final_line_match = list(re.finditer(r"(?im)^(?:\s*final\s*)$", cleaned))
    if final_line_match:
        cleaned = cleaned[final_line_match[-1].end() :]

    cleaned = re.sub(r"<\|[^|]+?\|>", "", cleaned)
    cleaned = _remove_prompt_echo(cleaned, prompt)
    for channel in ("analysis", "commentary", "final"):
        if cleaned.lower().startswith(channel):
            cleaned = cleaned[len(channel) :].lstrip(" \t\r\n:-")
            break

    answer_markers = list(re.finditer(r"(?i)\banswer\s*:\s*", cleaned))
    if answer_markers:
        cleaned = cleaned[answer_markers[-1].end() :]

    lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered in {"analysis", "commentary", "final", "assistant", "user"}:
            continue
        if re.match(
            r"(?i)^(we need|we should|need to|user asks|the user asks|as per instruction|answer should|provide\b)",
            stripped,
        ):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip(" \t\r\n:-")
    cleaned = _remove_prompt_echo(cleaned, prompt)
    return cleaned.strip()


def _assert_strict_runtime(model):
    stats = model.runtime_stats()
    expected = {
        "selective_fused_runtime": True,
        "adapter_name": "gpt_oss_mxfp4_reference",
        "mxfp4_execution": "reference_dequant",
        "dense_contains_full_fused_experts": False,
        "requires_dense_fallback": False,
        "strict_full_fused_tensor_loaded": False,
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise RuntimeError(f"strict runtime invariant failed: {key}={stats.get(key)!r}, expected {value!r}")

    unused = int(stats.get("unused_expert_builds_this_run") or 0)
    if unused != 0:
        raise RuntimeError(f"strict runtime invariant failed: unused_expert_builds_this_run={unused}, expected 0")

    return stats


def _build_model_kwargs(args):
    dense_device = args.dense_device or args.device
    return {
        "device": dense_device,
        "dtype": _dtype_from_name(args.dtype),
        "max_seq_len": args.max_seq_len,
        "prefetching": True,
        "moe_strict_streaming": True,
        "moe_allow_dense_fallback": False,
        "moe_expert_cache_mb": args.moe_expert_cache_mb,
        "moe_cpu_expert_cache_mb": args.cpu_cache_mb if args.moe_cpu_expert_cache_mb is None else args.moe_cpu_expert_cache_mb,
        "vram_cache_mode": args.vram_cache_mode,
        "vram_cache_mb": args.vram_cache_mb,
        "dense_expert_cache_mb": args.dense_expert_cache_mb,
        "dense_expert_cache_min_requests": args.dense_expert_cache_min_requests,
        "dense_expert_cache_max_builds_without_hit": args.dense_expert_cache_max_builds_without_hit,
        "os_reserved_ram_mb": args.os_reserved_ram_mb,
        "abort_unsafe_context": True,
        "resume_split_build": True,
        "lazy_build": True,
        "expert_materialization_mode": "direct_slice",
        "layer_cleanup_interval": args.layer_cleanup_interval,
        "mxfp4_device": args.mxfp4_device,
        "expert_matmul_device": args.expert_matmul_device,
        "mxfp4_execution": args.mxfp4_execution,
        "hf_triton_module_cache_mb": args.hf_triton_module_cache_mb,
        "hf_triton_module_cache_min_requests": args.hf_triton_module_cache_min_requests,
        "dense_device": args.dense_device,
        "max_vram_mb": args.max_vram_mb,
        "sync_cuda_timing": False,
        "enable_gpt_oss_batch_direct_slice": args.expert_execution_mode == "grouped_by_expert",
        "expert_execution_mode": args.expert_execution_mode,
        "keep_nontransformer_resident": True,
        "quiet_progress": args.quiet_progress,
    }


def _generate(model, input_ids, args, profiler=None):
    attention_mask = torch.ones_like(input_ids, device=model.device)
    generate_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": bool(args.use_kv_cache),
    }

    if not args.stream and not args.profile_speed:
        if profiler is not None:
            profiler.enable()
        try:
            return model.generate(input_ids, **generate_kwargs), "", []
        finally:
            if profiler is not None:
                profiler.disable()

    streamer = TimingTextIteratorStreamer(
        model.tokenizer,
        skip_prompt=True,
        skip_special_tokens=not args.use_harmony_prompt,
        timeout=5.0,
    )
    generate_kwargs["streamer"] = streamer
    result = {}
    stream_buffer = ""
    streamed_text = ""
    printed_chars = 0

    def _visible_stream_text(buffer):
        if not args.use_harmony_prompt:
            return re.sub(r"<\|[^|]+?\|>", "", buffer)
        for marker in HARMONY_FINAL_MARKERS:
            if marker in buffer:
                visible = buffer.rsplit(marker, 1)[-1]
                visible = re.sub(r"<\|[^|]+?\|>", "", visible)
                return visible
        return ""

    def _target():
        try:
            if profiler is not None:
                profiler.enable()
            result["output"] = model.generate(input_ids, **generate_kwargs)
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            if profiler is not None:
                profiler.disable()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    while True:
        try:
            chunk = next(streamer)
        except queue.Empty:
            if "error" in result:
                raise result["error"]
            if not thread.is_alive():
                break
            continue
        except StopIteration:
            break
        stream_buffer += chunk
        visible_text = _visible_stream_text(stream_buffer)
        if args.stream and len(visible_text) > printed_chars:
            delta = visible_text[printed_chars:]
            print(delta, end="", flush=True)
            streamed_text += delta
            printed_chars = len(visible_text)
    thread.join()

    if "error" in result:
        raise result["error"]
    return result["output"], streamed_text, streamer.token_events


def _compact_stats(stats, decode_elapsed, generated_tokens, args):
    denom = max(1, int(generated_tokens))
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "generated_tokens": int(generated_tokens),
        "seconds_per_token": decode_elapsed / denom,
        "tokens_per_second": int(generated_tokens) / decode_elapsed if decode_elapsed > 0 else 0.0,
        "peak_vram_mb": peak_vram_mb,
        "dense_fallback_used": bool(stats.get("requires_dense_fallback")),
        "strict_full_fused_tensor_loaded": bool(stats.get("strict_full_fused_tensor_loaded")),
        "unused_expert_builds_this_run": int(stats.get("unused_expert_builds_this_run") or 0),
        "use_kv_cache": bool(args.use_kv_cache),
        "expert_execution_mode": args.expert_execution_mode,
        "mxfp4_execution": args.mxfp4_execution,
        "runtime_mxfp4_execution": stats.get("runtime_mxfp4_execution"),
        "run_expert_calls": int(stats.get("run_expert_calls") or 0),
        "run_expert_calls_per_generated_token": (int(stats.get("run_expert_calls") or 0) / denom),
        "grouped_expert_calls": int(stats.get("grouped_expert_calls") or 0),
        "per_token_expert_calls": int(stats.get("per_token_expert_calls") or 0),
        "vram_cache_mode": args.vram_cache_mode,
        "gpu_packed_expert_cache_hits": int(stats.get("gpu_packed_expert_cache_hits") or 0),
        "gpu_packed_expert_cache_misses": int(stats.get("gpu_packed_expert_cache_misses") or 0),
        "gpu_dense_expert_cache_hits": int(stats.get("gpu_dense_expert_cache_hits") or 0),
        "gpu_dense_expert_cache_misses": int(stats.get("gpu_dense_expert_cache_misses") or 0),
        "gpu_dense_expert_builds": int(stats.get("gpu_dense_expert_builds") or 0),
        "gpu_dense_expert_build_seconds": float(stats.get("gpu_dense_expert_build_seconds") or 0.0),
        "gpu_dense_expert_admission_skips": int(stats.get("gpu_dense_expert_admission_skips") or 0),
        "gpu_dense_expert_unique_requests": int(stats.get("gpu_dense_expert_unique_requests") or 0),
        "gpu_dense_expert_min_requests": int(stats.get("gpu_dense_expert_min_requests") or 0),
        "gpu_dense_expert_max_builds_without_hit": int(stats.get("gpu_dense_expert_max_builds_without_hit") or 0),
        "gpu_dense_expert_disabled_no_hits": bool(stats.get("gpu_dense_expert_disabled_no_hits")),
        "cpu_packed_expert_cache_hits": int(stats.get("cpu_expert_cache_hits") or 0),
        "cpu_packed_expert_cache_misses": int(stats.get("cpu_expert_cache_misses") or 0),
        "dense_layer_cache_hits": int(stats.get("dense_layer_cache_hits") or 0),
        "dense_layer_cache_misses": int(stats.get("dense_layer_cache_misses") or 0),
        "cache_bytes_avoided": int(stats.get("gpu_packed_expert_bytes_avoided") or 0)
        + int(stats.get("gpu_dense_expert_bytes_avoided") or 0),
        "gpu_packed_expert_cache_mb": (int(stats.get("gpu_packed_expert_bytes") or 0) / 1024**2),
        "gpu_dense_expert_cache_mb": (int(stats.get("gpu_dense_expert_bytes") or 0) / 1024**2),
        "hf_triton_kernel_seconds": float(stats.get("hf_triton_kernel_seconds") or 0.0),
        "hf_triton_module_cache_hits": int(stats.get("hf_triton_module_cache_hits") or 0),
        "hf_triton_module_cache_misses": int(stats.get("hf_triton_module_cache_misses") or 0),
        "hf_triton_module_builds": int(stats.get("hf_triton_module_builds") or 0),
        "hf_triton_module_build_seconds": float(stats.get("hf_triton_module_build_seconds") or 0.0),
        "hf_triton_module_cache_mb": (int(stats.get("hf_triton_module_cache_bytes") or 0) / 1024**2),
        "hf_triton_module_cache_evictions": int(stats.get("hf_triton_module_cache_evictions") or 0),
        "hf_triton_module_cache_min_requests": int(stats.get("hf_triton_module_cache_min_requests") or 0),
        "hf_triton_module_admission_skips": int(stats.get("hf_triton_module_admission_skips") or 0),
        "hf_triton_module_unique_requests": int(stats.get("hf_triton_module_unique_requests") or 0),
        "hf_triton_module_direct_loads_skipped": int(stats.get("hf_triton_module_direct_loads_skipped") or 0),
        "hf_triton_fallbacks": int(stats.get("hf_triton_fallbacks") or 0),
    }


def _sum_layers(layer_breakdown, key):
    return sum(float(item.get(key) or 0.0) for item in layer_breakdown.values())


def _top_functions(profiler, limit=10):
    if profiler is None:
        return []
    stats = pstats.Stats(profiler, stream=io.StringIO())
    rows = []
    for func, values in stats.stats.items():
        cc, nc, tt, ct, _callers = values
        filename, line, name = func
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}:{name}",
                "file": filename,
                "line": line,
                "primitive_calls": cc,
                "calls": nc,
                "total_seconds": tt,
                "cumulative_seconds": ct,
            }
        )
    return sorted(rows, key=lambda item: item["cumulative_seconds"], reverse=True)[:limit]


def _top_layers(layer_breakdown, limit=10):
    rows = []
    for layer, item in layer_breakdown.items():
        rows.append(
            {
                "layer": layer,
                "layer_loop_seconds": float(item.get("layer_loop_time") or 0.0),
                "layer_forward_seconds": float(item.get("layer_forward_time") or 0.0),
                "mlp_total_seconds": float(item.get("mlp_total_time") or 0.0),
                "attention_seconds": float(item.get("attention_time") or 0.0),
                "router_seconds": float(item.get("router_time") or 0.0),
                "direct_slice_seconds": float(item.get("direct_slice_read_time") or 0.0),
                "mxfp4_dequant_seconds": float(item.get("mxfp4_dequant_time") or 0.0),
                "expert_python_overhead_seconds": float(item.get("expert_python_overhead_time") or 0.0),
            }
        )
    return sorted(rows, key=lambda item: item["layer_loop_seconds"], reverse=True)[:limit]


def _token_timing_profile(token_events, decode_started, decode_elapsed):
    token_durations = []
    previous = decode_started
    for event in token_events:
        timestamp = float(event["timestamp"])
        token_count = max(1, int(event.get("token_count") or 1))
        duration = max(0.0, timestamp - previous)
        token_durations.extend([duration / token_count] * token_count)
        previous = timestamp

    if not token_durations:
        return {
            "first_token_seconds": decode_elapsed,
            "token_2_plus_average_seconds": None,
            "average_per_token_excluding_first_seconds": None,
            "token_event_count": 0,
        }

    tail = token_durations[1:]
    tail_average = (sum(tail) / len(tail)) if tail else None
    return {
        "first_token_seconds": token_durations[0],
        "token_2_plus_average_seconds": tail_average,
        "average_per_token_excluding_first_seconds": tail_average,
        "token_event_count": len(token_durations),
    }


def _build_speed_profile(
    *,
    stats,
    decode_elapsed,
    generated_tokens,
    token_events,
    decode_started,
    tokenizer_decode_cleaning_seconds,
    profiler,
):
    layer_breakdown = _merge_layer_timing_breakdown(stats)
    lm_head = layer_breakdown.get("lm_head", {})
    layer_loop_seconds = float(stats.get("layer_loop_seconds") or 0.0)
    layer_forward_seconds = float(stats.get("layer_forward_seconds") or 0.0)
    mlp_total_seconds = _sum_layers(layer_breakdown, "mlp_total_time")
    direct_slice_seconds = float(stats.get("selected_expert_materialization_time_this_run") or 0.0)
    mxfp4_dequant_seconds = float(stats.get("mxfp4_dequant_seconds") or 0.0)
    hf_triton_kernel_seconds = float(stats.get("hf_triton_kernel_seconds") or 0.0)
    gate_up_seconds = float(stats.get("gate_up_matmul_seconds") or 0.0)
    down_seconds = float(stats.get("down_matmul_seconds") or 0.0)
    activation_seconds = float(stats.get("activation_seconds") or 0.0)
    input_cast_seconds = float((stats.get("fused_adapter_stats", {}) or {}).get("input_cast_seconds") or 0.0)
    attention_seconds = _sum_layers(layer_breakdown, "attention_time")
    router_seconds = _sum_layers(layer_breakdown, "router_time")
    dense_load_seconds = _sum_layers(layer_breakdown, "dense_load_time")
    move_seconds = _sum_layers(layer_breakdown, "move_to_device_time")
    cleanup_seconds = float(stats.get("cleanup_seconds") or 0.0)
    lm_head_seconds = float(lm_head.get("layer_loop_time") or lm_head.get("layer_forward_time") or 0.0)
    non_layer_generate_seconds = max(0.0, decode_elapsed - layer_loop_seconds)
    expert_known = (
        direct_slice_seconds
        + mxfp4_dequant_seconds
        + hf_triton_kernel_seconds
        + gate_up_seconds
        + down_seconds
        + activation_seconds
        + input_cast_seconds
    )
    expert_aggregation_seconds = max(0.0, mlp_total_seconds - expert_known)

    raw_cumulative_breakdown = {
        "layer_dense_load_seconds": dense_load_seconds,
        "layer_move_to_device_seconds": move_seconds,
        "attention_seconds": attention_seconds,
        "router_seconds": router_seconds,
        "selected_expert_direct_slice_seconds": direct_slice_seconds,
        "packed_mxfp4_transfer_seconds": 0.0,
        "mxfp4_unpack_dequant_seconds": mxfp4_dequant_seconds,
        "hf_triton_kernel_seconds": hf_triton_kernel_seconds,
        "gate_up_matmul_seconds": gate_up_seconds,
        "activation_seconds": activation_seconds,
        "down_matmul_seconds": down_seconds,
        "expert_aggregation_seconds": expert_aggregation_seconds,
        "cache_lookup_seconds": 0.0,
        "cleanup_seconds": cleanup_seconds,
        "sampling_logits_lm_head_seconds": lm_head_seconds + non_layer_generate_seconds,
        "tokenizer_decode_output_cleaning_seconds": tokenizer_decode_cleaning_seconds,
    }
    raw_cumulative_seconds = sum(raw_cumulative_breakdown.values())
    if raw_cumulative_seconds > 0:
        scale = min(1.0, decode_elapsed / raw_cumulative_seconds)
        categories = {key: value * scale for key, value in raw_cumulative_breakdown.items()}
    else:
        scale = 1.0
        categories = raw_cumulative_breakdown.copy()
    accounted_before_residual = sum(categories.values())
    python_overhead_seconds = max(0.0, decode_elapsed - accounted_before_residual)
    categories["python_overhead_unaccounted_seconds"] = python_overhead_seconds
    accounted_seconds = sum(categories.values())
    coverage = accounted_seconds / decode_elapsed if decode_elapsed > 0 else 0.0
    unaccounted_time = max(0.0, decode_elapsed - accounted_seconds)
    overlap_seconds = max(0.0, raw_cumulative_seconds - decode_elapsed)

    return {
        "decode_seconds": decode_elapsed,
        "generated_tokens": int(generated_tokens),
        "seconds_per_token": decode_elapsed / max(1, int(generated_tokens)),
        "timing_coverage_ratio": coverage,
        "unaccounted_time_seconds": unaccounted_time if coverage < 0.90 else 0.0,
        "time_per_token_breakdown": categories,
        "raw_cumulative_timing_breakdown": raw_cumulative_breakdown,
        "raw_cumulative_seconds": raw_cumulative_seconds,
        "overlapped_or_double_counted_seconds": overlap_seconds,
        "exclusive_breakdown_scale_factor": scale,
        "per_token_timing": _token_timing_profile(token_events, decode_started, decode_elapsed),
        "top_10_slowest_functions": _top_functions(profiler, limit=10),
        "top_10_slowest_layers": _top_layers(layer_breakdown, limit=10),
        "instrumentation_notes": [
            "packed_mxfp4_transfer_seconds is not independently timed by the reference path; transfer into CUDA is currently included in mxfp4_unpack_dequant_seconds.",
            "cache_lookup_seconds is not independently timed by the current selected-expert adapter; cache hit/miss counts are still reported in runtime_stats.",
            "expert_aggregation_seconds is derived from mlp_total minus directly measured expert work, so it includes index_add, routing-weight application, and residual adapter loop overhead.",
            "raw_cumulative_timing_breakdown can exceed decode_seconds because dense prefetch, per-expert materialization, CUDA work, and thread waits overlap or nest.",
            "time_per_token_breakdown is normalized to exclusive decode wall time when raw cumulative timings overlap.",
            "python_overhead_unaccounted_seconds is the residual needed to reconcile exclusive category totals to decode_seconds.",
        ],
        "runtime_stats_summary": {
            "expert_cache_hits": stats.get("expert_cache_hits", 0),
            "expert_cache_misses": stats.get("expert_cache_misses", 0),
            "gpu_packed_expert_cache_hits": stats.get("gpu_packed_expert_cache_hits", 0),
            "gpu_packed_expert_cache_misses": stats.get("gpu_packed_expert_cache_misses", 0),
            "gpu_packed_expert_bytes": stats.get("gpu_packed_expert_bytes", 0),
            "gpu_packed_expert_bytes_avoided": stats.get("gpu_packed_expert_bytes_avoided", 0),
            "gpu_dense_expert_cache_hits": stats.get("gpu_dense_expert_cache_hits", 0),
            "gpu_dense_expert_cache_misses": stats.get("gpu_dense_expert_cache_misses", 0),
            "gpu_dense_expert_bytes": stats.get("gpu_dense_expert_bytes", 0),
            "gpu_dense_expert_bytes_avoided": stats.get("gpu_dense_expert_bytes_avoided", 0),
            "gpu_dense_expert_builds": stats.get("gpu_dense_expert_builds", 0),
            "gpu_dense_expert_build_seconds": stats.get("gpu_dense_expert_build_seconds", 0.0),
            "gpu_dense_expert_admission_skips": stats.get("gpu_dense_expert_admission_skips", 0),
            "gpu_dense_expert_unique_requests": stats.get("gpu_dense_expert_unique_requests", 0),
            "gpu_dense_expert_min_requests": stats.get("gpu_dense_expert_min_requests", 0),
            "gpu_dense_expert_max_builds_without_hit": stats.get("gpu_dense_expert_max_builds_without_hit", 0),
            "gpu_dense_expert_disabled_no_hits": stats.get("gpu_dense_expert_disabled_no_hits", False),
            "cpu_expert_cache_hits": stats.get("cpu_expert_cache_hits", 0),
            "cpu_expert_cache_misses": stats.get("cpu_expert_cache_misses", 0),
            "dense_layer_cache_hits": stats.get("dense_layer_cache_hits", 0),
            "dense_layer_cache_misses": stats.get("dense_layer_cache_misses", 0),
            "packed_expert_bytes_loaded": stats.get("packed_expert_bytes_loaded", 0),
            "run_expert_calls": stats.get("run_expert_calls", 0),
            "run_expert_calls_per_generated_token": (int(stats.get("run_expert_calls") or 0) / max(1, int(generated_tokens))),
            "grouped_expert_calls": stats.get("grouped_expert_calls", 0),
            "per_token_expert_calls": stats.get("per_token_expert_calls", 0),
            "expert_input_rows": stats.get("expert_input_rows", 0),
            "direct_slice_expert_loads_this_run": stats.get("direct_slice_expert_loads_this_run", 0),
            "direct_slice_packed_bytes_read_this_run": stats.get("direct_slice_packed_bytes_read_this_run", 0),
            "layer_forward_seconds": layer_forward_seconds,
            "layer_loop_seconds": layer_loop_seconds,
        },
    }


def run(args):
    _validate_device_args(args)
    if not args.confirm_120b:
        raise SystemExit("This runner is intended for strict GPT-OSS 120B use and requires --confirm-120b.")

    config = load_config(args.model, cache_dir=args.hf_cache_dir, token=args.hf_token)
    if not _is_gpt_oss_config(config):
        raise SystemExit("The supplied model config does not look like GPT-OSS.")
    if _is_probably_120b(args, config) and not args.confirm_120b:
        raise SystemExit("GPT-OSS 120B runs require --confirm-120b.")

    print(json.dumps({"startup_runtime_environment": _runtime_environment(args)}), file=sys.stderr)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model_kwargs = _build_model_kwargs(args)
    if args.hf_token:
        model_kwargs["hf_token"] = args.hf_token

    model = AutoModel.from_pretrained(args.model, **model_kwargs)
    _assert_strict_runtime(model)

    input_ids, warning = _build_input_ids(
        model,
        args.prompt,
        use_harmony_prompt=args.use_harmony_prompt,
        reasoning_effort=args.reasoning_effort,
    )
    if warning:
        print(warning, file=sys.stderr)

    input_ids = input_ids.to(model.device)
    input_token_count = int(input_ids.shape[-1])
    if input_token_count > args.max_seq_len:
        raise SystemExit(f"prompt tokens {input_token_count} exceed --max-seq-len {args.max_seq_len}")
    if input_token_count + int(args.max_new_tokens) > args.max_seq_len:
        raise SystemExit(
            f"prompt tokens {input_token_count} + max_new_tokens {args.max_new_tokens} exceed --max-seq-len {args.max_seq_len}"
        )

    if hasattr(model, "clear_runtime_caches"):
        model.clear_runtime_caches()

    profiler = cProfile.Profile() if args.profile_speed else None
    decode_started = time.perf_counter()
    output, streamed_text, token_events = _generate(model, input_ids, args, profiler=profiler)
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.synchronize(model.device)
    decode_elapsed = time.perf_counter() - decode_started

    if hasattr(model, "flush_runtime_progress"):
        model.flush_runtime_progress()

    stats = _assert_strict_runtime(model)
    text_started = time.perf_counter()
    raw_generated_text, generated_tokens = _decode_new_tokens(model, input_ids, output)
    clean_answer = _clean_generated_text(raw_generated_text, args.prompt)
    tokenizer_decode_cleaning_seconds = time.perf_counter() - text_started

    if streamed_text:
        if not streamed_text.endswith("\n"):
            print()
    else:
        print(clean_answer)
    print(
        json.dumps(
            {"run_summary": _compact_stats(stats, decode_elapsed, generated_tokens, args)},
            indent=2,
        ),
        file=sys.stderr,
    )
    if args.profile_speed:
        profile = _build_speed_profile(
            stats=stats,
            decode_elapsed=decode_elapsed,
            generated_tokens=generated_tokens,
            token_events=token_events,
            decode_started=decode_started,
            tokenizer_decode_cleaning_seconds=tokenizer_decode_cleaning_seconds,
            profiler=profiler,
        )
        profile_path = Path(args.profile_json)
        if profile_path.parent != Path("."):
            profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "speed_profile": {
                        "path": str(profile_path),
                        "timing_coverage_ratio": profile["timing_coverage_ratio"],
                        "unaccounted_time_seconds": profile["unaccounted_time_seconds"],
                        "first_token_seconds": profile["per_token_timing"]["first_token_seconds"],
                        "token_2_plus_average_seconds": profile["per_token_timing"]["token_2_plus_average_seconds"],
                        "top_10_slowest_functions": profile["top_10_slowest_functions"],
                        "top_10_slowest_layers": profile["top_10_slowest_layers"],
                    }
                },
                indent=2,
            ),
            file=sys.stderr,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Run GPT-OSS with strict direct-slice streaming and clean output.")
    parser.add_argument("model", help="Local GPT-OSS model path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--use-kv-cache", dest="use_kv_cache", action="store_true", default=True)
    parser.add_argument("--no-kv-cache", dest="use_kv_cache", action="store_false")
    parser.add_argument("--use-harmony-prompt", action="store_true")
    parser.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "high"))
    parser.add_argument("--stream", dest="stream", action="store_true", default=True)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dense-device")
    parser.add_argument("--mxfp4-device", default="cuda")
    parser.add_argument("--expert-matmul-device", default="cuda")
    parser.add_argument("--mxfp4-execution", choices=("reference_cuda", "hf_triton", "triton_fused"), default="reference_cuda")
    parser.add_argument(
        "--hf-triton-module-cache-mb",
        type=int,
        default=0,
        help="Global VRAM budget for cached HF Triton one-expert modules.",
    )
    parser.add_argument(
        "--hf-triton-module-cache-min-requests",
        type=int,
        default=2,
        help="Only build/cache an HF Triton selected-expert module after this many requests for the same layer/expert.",
    )
    parser.add_argument("--max-vram-mb", type=int, default=7600)
    parser.add_argument("--layer-cleanup-interval", type=int, default=8)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--moe-expert-cache-mb", type=int, default=1024)
    parser.add_argument("--moe-cpu-expert-cache-mb", type=int)
    parser.add_argument("--vram-cache-mode", choices=("off", "packed_experts", "dense_layers", "hybrid"), default="off")
    parser.add_argument("--vram-cache-mb", type=int, default=4000)
    parser.add_argument(
        "--dense-expert-cache-mb",
        type=int,
        help="VRAM budget for selected-expert predequantized dense cache. Defaults to --vram-cache-mb for dense_layers mode and 0 for hybrid.",
    )
    parser.add_argument(
        "--dense-expert-cache-min-requests",
        type=int,
        default=2,
        help="Only build a dense selected-expert cache entry after this many requests for the same layer/expert.",
    )
    parser.add_argument(
        "--dense-expert-cache-max-builds-without-hit",
        type=int,
        default=256,
        help="Disable dense selected-expert admission after this many builds if there have been no dense cache hits. Set 0 to disable the cutoff.",
    )
    parser.add_argument("--cpu-cache-mb", type=int, default=12000)
    parser.add_argument("--expert-execution-mode", choices=("per_token", "grouped_by_expert"), default="grouped_by_expert")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--os-reserved-ram-mb", type=int, default=8192)
    parser.add_argument("--hf-token")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--confirm-120b", action="store_true")
    parser.add_argument("--profile-speed", action="store_true")
    parser.add_argument("--profile-json", default="speed_profile.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"status": "clean_abort", "reason": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
