"""Benchmark one GPT-OSS MXFP4 selected expert.

This is intentionally isolated from the production runtime. It loads one
selected expert from GPT-OSS safetensors, benchmarks the current reference path,
then compares it with a conservative dequant-chunked down-projection prototype.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
AIR_LLM_ROOT = REPO_ROOT / "air_llm"
if str(AIR_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(AIR_LLM_ROOT))

import torch  # noqa: E402

from airllm.gpt_oss_mxfp4 import (  # noqa: E402
    dequantize_mxfp4_expert,
    dequantize_mxfp4_projection,
    mxfp4_state_nbytes,
    run_gpt_oss_selected_expert_reference_timed,
    tensor_nbytes,
)
from airllm.moe_layout_probe import (  # noqa: E402
    GPT_OSS_PACKED_EXPERT_SUFFIXES,
    load_config,
    load_safetensors_index,
)


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = (name or "").lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _load_one_expert_from_safetensors(model_path: Path, layer_id: int, expert_id: int) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required for this benchmark") from exc

    index = load_safetensors_index(str(model_path))
    if not index:
        raise FileNotFoundError("model.safetensors.index.json or local safetensors shards are required")
    weight_map = index.get("weight_map") or {}
    layer_prefix = f"model.layers.{int(layer_id)}"
    tensor_names = [layer_prefix + suffix for suffix in GPT_OSS_PACKED_EXPERT_SUFFIXES]
    missing = [name for name in tensor_names if name not in weight_map]
    if missing:
        raise KeyError(f"missing GPT-OSS packed tensors for {layer_prefix}: {missing}")

    expert_state: Dict[str, torch.Tensor] = {}
    tensor_shapes: Dict[str, list[int]] = {}
    shard_names = sorted({weight_map[name] for name in tensor_names})
    for name in tensor_names:
        shard_path = model_path / weight_map[name]
        if not shard_path.exists():
            raise FileNotFoundError(f"missing local safetensors shard: {shard_path}")
        mapped_name = name.split(".mlp.experts.", 1)[1]
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(name)
            tensor_shapes[name] = list(tensor_slice.get_shape())
            expert_state[mapped_name] = tensor_slice[int(expert_id)].contiguous()

    metadata = {
        "model": str(model_path),
        "layer_id": int(layer_id),
        "expert_id": int(expert_id),
        "shards": shard_names,
        "tensor_shapes": tensor_shapes,
        "selected_expert_shapes": {key: list(value.shape) for key, value in expert_state.items()},
        "packed_expert_bytes": mxfp4_state_nbytes(expert_state),
    }
    return expert_state, metadata


def _move_expert_state(expert_state: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device=device, non_blocking=device.type == "cuda") for key, value in expert_state.items()}


def run_gpt_oss_selected_expert_down_chunked_timed(
    expert_state: Mapping[str, torch.Tensor],
    expert_input: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    alpha: float = 1.702,
    limit: float = 7.0,
    mxfp4_device: torch.device,
    matmul_device: torch.device,
    output_device: torch.device,
    down_output_chunk_size: int = 512,
    sync_cuda_timing: bool = True,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    if mxfp4_device != matmul_device:
        raise ValueError("chunked prototype currently requires mxfp4_device and matmul_device to match")
    device = matmul_device
    timings: Dict[str, Any] = {}

    maybe_sync = lambda: _sync(device) if sync_cuda_timing else None

    maybe_sync()
    started = time.perf_counter()
    gate_up_proj = dequantize_mxfp4_projection(
        expert_state["gate_up_proj_blocks"].to(device),
        expert_state["gate_up_proj_scales"].to(device),
        dtype=compute_dtype,
    )
    gate_up_bias = expert_state["gate_up_proj_bias"].to(device=device, dtype=compute_dtype)
    maybe_sync()
    timings["gate_up_dequant_seconds"] = time.perf_counter() - started

    maybe_sync()
    started = time.perf_counter()
    current_state = expert_input.to(device=device, dtype=compute_dtype)
    maybe_sync()
    timings["input_cast_seconds"] = time.perf_counter() - started

    maybe_sync()
    started = time.perf_counter()
    gate_up = current_state @ gate_up_proj + gate_up_bias
    maybe_sync()
    timings["gate_up_matmul_seconds"] = time.perf_counter() - started

    maybe_sync()
    started = time.perf_counter()
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    maybe_sync()
    timings["activation_seconds"] = time.perf_counter() - started

    down_blocks = expert_state["down_proj_blocks"]
    down_scales = expert_state["down_proj_scales"]
    down_bias = expert_state["down_proj_bias"]
    output_width = int(down_bias.numel())
    output = torch.empty(
        gated_output.shape[0],
        output_width,
        dtype=compute_dtype,
        device=device,
    )

    down_dequant_seconds = 0.0
    down_matmul_seconds = 0.0
    chunk_count = 0
    for start in range(0, output_width, max(1, int(down_output_chunk_size))):
        end = min(start + max(1, int(down_output_chunk_size)), output_width)
        chunk_count += 1

        maybe_sync()
        started = time.perf_counter()
        down_proj_chunk = dequantize_mxfp4_projection(
            down_blocks[start:end].to(device),
            down_scales[start:end].to(device),
            dtype=compute_dtype,
        )
        down_bias_chunk = down_bias[start:end].to(device=device, dtype=compute_dtype)
        maybe_sync()
        down_dequant_seconds += time.perf_counter() - started

        maybe_sync()
        started = time.perf_counter()
        output[:, start:end] = gated_output @ down_proj_chunk + down_bias_chunk
        maybe_sync()
        down_matmul_seconds += time.perf_counter() - started

    timings["down_dequant_seconds"] = down_dequant_seconds
    timings["down_matmul_seconds"] = down_matmul_seconds
    timings["mxfp4_dequant_seconds"] = timings["gate_up_dequant_seconds"] + down_dequant_seconds
    timings["expert_matmul_seconds"] = timings["gate_up_matmul_seconds"] + down_matmul_seconds
    timings["chunk_count"] = chunk_count
    chunk_width = min(max(1, int(down_output_chunk_size)), output_width)
    down_chunk_elements = int(chunk_width) * int(down_blocks.shape[1]) * int(down_blocks.shape[2]) * 2
    dtype_bytes = torch.tensor([], dtype=compute_dtype).element_size()
    timings["dequantized_temporary_bytes_estimate"] = tensor_nbytes(gate_up_proj) + down_chunk_elements * dtype_bytes
    return output.to(device=output_device, dtype=expert_input.dtype), timings


def predequantize_gpt_oss_selected_expert_timed(
    expert_state: Mapping[str, torch.Tensor],
    *,
    compute_dtype: torch.dtype,
    device: torch.device,
    sync_cuda_timing: bool = True,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if sync_cuda_timing:
        _sync(device)
    started = time.perf_counter()
    dense = dequantize_mxfp4_expert(expert_state, dtype=compute_dtype, device=device)
    if sync_cuda_timing:
        _sync(device)
    timings = {
        "predequant_seconds": time.perf_counter() - started,
        "dense_expert_bytes": sum(tensor_nbytes(tensor) for tensor in dense.values()),
    }
    return dense, timings


def run_gpt_oss_selected_expert_predequantized_timed(
    dense_expert: Mapping[str, torch.Tensor],
    expert_input: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    alpha: float = 1.702,
    limit: float = 7.0,
    matmul_device: torch.device,
    output_device: torch.device,
    sync_cuda_timing: bool = True,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    device = matmul_device
    timings: Dict[str, Any] = {}
    maybe_sync = lambda: _sync(device) if sync_cuda_timing else None

    maybe_sync()
    started = time.perf_counter()
    current_state = expert_input.to(device=device, dtype=compute_dtype)
    maybe_sync()
    timings["input_cast_seconds"] = time.perf_counter() - started

    maybe_sync()
    started = time.perf_counter()
    gate_up = current_state @ dense_expert["gate_up_proj"] + dense_expert["gate_up_proj_bias"]
    maybe_sync()
    timings["gate_up_matmul_seconds"] = time.perf_counter() - started

    maybe_sync()
    started = time.perf_counter()
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    maybe_sync()
    timings["activation_seconds"] = time.perf_counter() - started

    maybe_sync()
    started = time.perf_counter()
    output = gated_output @ dense_expert["down_proj"] + dense_expert["down_proj_bias"]
    output = output.to(device=output_device, dtype=expert_input.dtype)
    maybe_sync()
    timings["down_matmul_seconds"] = time.perf_counter() - started
    timings["mxfp4_dequant_seconds"] = 0.0
    timings["expert_matmul_seconds"] = timings["gate_up_matmul_seconds"] + timings["down_matmul_seconds"]
    return output, timings


class BenchmarkStageError(RuntimeError):
    def __init__(self, detail: Dict[str, Any]):
        self.detail = detail
        super().__init__(_exception_summary(detail))


def _exception_detail(stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
    }


def _exception_summary(detail: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not detail:
        return None
    message = str(detail.get("message") or "")
    if not message:
        message = "<empty>"
    return f"{detail.get('stage')}: {detail.get('type')}: {message}"


def _load_hf_gpt_oss_triton_kernel():
    try:
        from transformers.integrations.hub_kernels import get_kernel
        import transformers.integrations.mxfp4 as hf_mxfp4

        hf_mxfp4.triton_kernels_hub = get_kernel("kernels-community/gpt-oss-triton-kernels")
        return hf_mxfp4, None, None
    except BaseException as exc:  # noqa: BLE001 - dependency/compiler failures must be reported, not fatal.
        detail = _exception_detail("load_hf_gpt_oss_triton_kernel", exc)
        return None, _exception_summary(detail), detail


def build_hf_triton_one_expert_module(
    expert_state: Mapping[str, torch.Tensor],
    *,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[Any, Optional[str], Optional[Dict[str, Any]]]:
    hf_mxfp4, error, error_detail = _load_hf_gpt_oss_triton_kernel()
    if hf_mxfp4 is None:
        return None, error, error_detail

    config = types.SimpleNamespace(
        num_local_experts=1,
        intermediate_size=int(intermediate_size),
        hidden_size=int(hidden_size),
        swiglu_limit=7.0,
    )
    stage = "construct_hf_mxfp4_module"
    try:
        module = hf_mxfp4.Mxfp4GptOssExperts(config).to(device)
        stage = "copy_hf_mxfp4_biases"
        with torch.no_grad():
            module.gate_up_proj_bias.copy_(expert_state["gate_up_proj_bias"].to(device=device, dtype=torch.float32).unsqueeze(0))
            module.down_proj_bias.copy_(expert_state["down_proj_bias"].to(device=device, dtype=torch.float32).unsqueeze(0))

        stage = "swizzle_hf_mxfp4_gate_up"
        hf_mxfp4.swizzle_mxfp4_convertops(
            expert_state["gate_up_proj_blocks"].unsqueeze(0).to(device),
            expert_state["gate_up_proj_scales"].unsqueeze(0).to(device),
            module,
            "gate_up_proj",
            device,
            hf_mxfp4.triton_kernels_hub,
        )
        stage = "swizzle_hf_mxfp4_down"
        hf_mxfp4.swizzle_mxfp4_convertops(
            expert_state["down_proj_blocks"].unsqueeze(0).to(device),
            expert_state["down_proj_scales"].unsqueeze(0).to(device),
            module,
            "down_proj",
            device,
            hf_mxfp4.triton_kernels_hub,
        )
        module._airllm_hf_mxfp4 = hf_mxfp4
        return module, None, None
    except BaseException as exc:  # noqa: BLE001
        detail = _exception_detail(stage, exc)
        return None, _exception_summary(detail), detail


def run_hf_triton_one_expert_timed(
    module,
    expert_input: torch.Tensor,
    *,
    output_device: torch.device,
    sync_cuda_timing: bool = True,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    device = expert_input.device
    hf_mxfp4 = getattr(module, "_airllm_hf_mxfp4", None)
    if hf_mxfp4 is None:
        raise RuntimeError("HF GPT-OSS Triton kernel module was not attached during setup")

    try:
        RoutingData = hf_mxfp4.triton_kernels_hub.routing.RoutingData
        GatherIndx = hf_mxfp4.triton_kernels_hub.routing.GatherIndx
        ScatterIndx = hf_mxfp4.triton_kernels_hub.routing.ScatterIndx
        compute_expt_data_torch = hf_mxfp4.triton_kernels_hub.routing.compute_expt_data_torch

        token_count = int(expert_input.shape[0])
        source_indices = torch.arange(token_count, device=device, dtype=torch.int32)
        hist = torch.tensor([token_count], device=device, dtype=torch.int32)
        gate_scal = torch.ones(token_count, device=device, dtype=torch.float32)
        routing_data = RoutingData(
            gate_scal,
            hist,
            1,
            1,
            compute_expt_data_torch(hist, 1, token_count),
        )
        gather_idx = GatherIndx(src_indx=source_indices, dst_indx=source_indices)
        scatter_idx = ScatterIndx(src_indx=source_indices, dst_indx=source_indices)
    except BaseException as exc:  # noqa: BLE001
        raise BenchmarkStageError(_exception_detail("build_hf_triton_routing_metadata", exc)) from exc

    if sync_cuda_timing:
        _sync(device)
    started = time.perf_counter()
    try:
        output = module(expert_input, routing_data, gather_idx, scatter_idx)
    except BaseException as exc:  # noqa: BLE001
        raise BenchmarkStageError(_exception_detail("hf_triton_module_forward", exc)) from exc
    if sync_cuda_timing:
        _sync(device)
    return output.to(device=output_device, dtype=expert_input.dtype), {
        "hf_triton_kernel_seconds": time.perf_counter() - started,
        "routing_rows": token_count,
    }


def _benchmark_variant(name, fn, *, warmup: int, iterations: int, device: torch.device) -> Dict[str, Any]:
    for _ in range(max(0, int(warmup))):
        fn()
    _sync(device)
    _reset_cuda_peak(device)
    timings = []
    last_output = None
    last_detail: Dict[str, Any] = {}
    for _ in range(max(1, int(iterations))):
        _sync(device)
        started = time.perf_counter()
        last_output, last_detail = fn()
        _sync(device)
        timings.append(time.perf_counter() - started)
    return {
        "name": name,
        "iterations": int(iterations),
        "average_seconds": sum(timings) / len(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "peak_vram_mb": _cuda_peak_mb(device),
        "last_detail": last_detail,
        "output": last_output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark one GPT-OSS MXFP4 selected expert.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mxfp4-device", default="cuda")
    parser.add_argument("--expert-matmul-device", default="cuda")
    parser.add_argument("--packed-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--down-output-chunk-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument(
        "--include-predequantized",
        action="store_true",
        help="Measure an upper-bound hot dense expert cache path. This is not strict-runtime integration.",
    )
    parser.add_argument(
        "--include-hf-triton",
        action="store_true",
        help="Try the Hugging Face/kernels-community GPT-OSS MXFP4 Triton path for one selected expert.",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model directory not found: {model_path}")

    config = load_config(str(model_path))
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config.get("intermediate_size", hidden_size))
    compute_dtype = _dtype_from_name(args.dtype)
    input_device = torch.device(args.device)
    mxfp4_device = torch.device(args.mxfp4_device)
    matmul_device = torch.device(args.expert_matmul_device)
    packed_device = torch.device(args.packed_device)
    if packed_device.type == "cuda":
        packed_device = mxfp4_device

    expert_state_cpu, metadata = _load_one_expert_from_safetensors(model_path, args.layer_id, args.expert_id)
    expert_state = _move_expert_state(expert_state_cpu, packed_device)
    expert_input = torch.randn(
        int(args.batch_size),
        hidden_size,
        device=input_device,
        dtype=compute_dtype,
    )

    def reference_fn():
        return run_gpt_oss_selected_expert_reference_timed(
            expert_state,
            expert_input,
            compute_dtype=compute_dtype,
            mxfp4_device=mxfp4_device,
            matmul_device=matmul_device,
            output_device=input_device,
            sync_cuda_timing=True,
        )

    def chunked_fn():
        return run_gpt_oss_selected_expert_down_chunked_timed(
            expert_state,
            expert_input,
            compute_dtype=compute_dtype,
            mxfp4_device=mxfp4_device,
            matmul_device=matmul_device,
            output_device=input_device,
            down_output_chunk_size=args.down_output_chunk_size,
            sync_cuda_timing=True,
        )

    reference = _benchmark_variant(
        "reference_cuda",
        reference_fn,
        warmup=args.warmup,
        iterations=args.iterations,
        device=matmul_device,
    )
    chunked = _benchmark_variant(
        "down_chunked",
        chunked_fn,
        warmup=args.warmup,
        iterations=args.iterations,
        device=matmul_device,
    )

    ref_output = reference.pop("output")
    chunked_output = chunked.pop("output")
    predequantized = None
    predequantized_correctness = None
    hf_triton = None
    hf_triton_correctness = None
    hf_triton_error = None
    hf_triton_error_detail = None
    if args.include_predequantized:
        dense_expert, predequant_detail = predequantize_gpt_oss_selected_expert_timed(
            expert_state,
            compute_dtype=compute_dtype,
            device=matmul_device,
            sync_cuda_timing=True,
        )

        def predequantized_fn():
            return run_gpt_oss_selected_expert_predequantized_timed(
                dense_expert,
                expert_input,
                compute_dtype=compute_dtype,
                matmul_device=matmul_device,
                output_device=input_device,
                sync_cuda_timing=True,
            )

        predequantized = _benchmark_variant(
            "predequantized_hot_expert",
            predequantized_fn,
            warmup=args.warmup,
            iterations=args.iterations,
            device=matmul_device,
        )
        pre_output = predequantized.pop("output")
        predequantized["predequant_detail"] = predequant_detail
        try:
            torch.testing.assert_close(pre_output, ref_output, rtol=args.rtol, atol=args.atol)
            predequantized_correctness = {
                "outputs_close": True,
                "max_abs_diff": float((ref_output - pre_output).abs().max().item()),
                "error": None,
            }
        except AssertionError as exc:
            predequantized_correctness = {
                "outputs_close": False,
                "max_abs_diff": float((ref_output - pre_output).abs().max().item()),
                "error": str(exc).splitlines()[0],
            }

    if args.include_hf_triton:
        module, hf_triton_error, hf_triton_error_detail = build_hf_triton_one_expert_module(
            expert_state,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=matmul_device,
        )
        if module is not None:
            def hf_triton_fn():
                return run_hf_triton_one_expert_timed(
                    module,
                    expert_input,
                    output_device=input_device,
                    sync_cuda_timing=True,
                )

            try:
                hf_triton = _benchmark_variant(
                    "hf_triton_fused",
                    hf_triton_fn,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=matmul_device,
                )
                hf_output = hf_triton.pop("output")
                try:
                    torch.testing.assert_close(hf_output, ref_output, rtol=args.rtol, atol=args.atol)
                    hf_triton_correctness = {
                        "outputs_close": True,
                        "max_abs_diff": float((ref_output - hf_output).abs().max().item()),
                        "error": None,
                    }
                except AssertionError as exc:
                    hf_triton_correctness = {
                        "outputs_close": False,
                        "max_abs_diff": float((ref_output - hf_output).abs().max().item()),
                        "error": str(exc).splitlines()[0],
                    }
            except BenchmarkStageError as exc:
                hf_triton_error_detail = exc.detail
                hf_triton_error = _exception_summary(hf_triton_error_detail)
            except BaseException as exc:  # noqa: BLE001
                hf_triton_error_detail = _exception_detail("benchmark_hf_triton_fused", exc)
                hf_triton_error = _exception_summary(hf_triton_error_detail)
    max_abs_diff = float((ref_output - chunked_output).abs().max().item())
    try:
        torch.testing.assert_close(chunked_output, ref_output, rtol=args.rtol, atol=args.atol)
        correct = True
        correctness_error = None
    except AssertionError as exc:
        correct = False
        correctness_error = str(exc).splitlines()[0]

    result = {
        "benchmark": "gpt_oss_one_selected_expert_mxfp4",
        "metadata": metadata,
        "config": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "batch_size": int(args.batch_size),
            "compute_dtype": str(compute_dtype),
            "input_device": str(input_device),
            "mxfp4_device": str(mxfp4_device),
            "expert_matmul_device": str(matmul_device),
            "packed_device": str(packed_device),
            "down_output_chunk_size": int(args.down_output_chunk_size),
            "warmup": int(args.warmup),
            "iterations": int(args.iterations),
        },
        "reference_cuda": reference,
        "down_chunked": chunked,
        "predequantized_hot_expert": predequantized,
        "predequantized_correctness": predequantized_correctness,
        "hf_triton_fused": hf_triton,
        "hf_triton_correctness": hf_triton_correctness,
        "hf_triton_error": hf_triton_error,
        "hf_triton_error_detail": hf_triton_error_detail,
        "correctness": {
            "outputs_close": correct,
            "max_abs_diff": max_abs_diff,
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "error": correctness_error,
        },
        "speedup": {
            "down_chunked_vs_reference": reference["average_seconds"] / chunked["average_seconds"]
            if chunked["average_seconds"] > 0 else None,
            "predequantized_vs_reference": (
                reference["average_seconds"] / predequantized["average_seconds"]
                if predequantized and predequantized["average_seconds"] > 0 else None
            ),
            "hf_triton_vs_reference": (
                reference["average_seconds"] / hf_triton["average_seconds"]
                if hf_triton and hf_triton["average_seconds"] > 0 else None
            ),
            "integrate_candidate": bool(correct and chunked["average_seconds"] < reference["average_seconds"]),
            "dense_hot_cache_candidate": bool(
                predequantized
                and predequantized_correctness
                and predequantized_correctness["outputs_close"]
                and predequantized["average_seconds"] < reference["average_seconds"]
            ),
            "hf_triton_integrate_candidate": bool(
                hf_triton
                and hf_triton_correctness
                and hf_triton_correctness["outputs_close"]
                and hf_triton["average_seconds"] < reference["average_seconds"]
            ),
        },
    }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_path:
        Path(args.json_path).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
