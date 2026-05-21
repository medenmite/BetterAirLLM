from __future__ import annotations

import types
import time
from typing import Any, Dict, Mapping, Optional

import torch


FP4_VALUES = (
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_FP4_LUT_CACHE = {}
_HF_GPT_OSS_TRITON_KERNEL = None


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _fp4_lut(dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    key = (dtype, device.type, device.index)
    cached = _FP4_LUT_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(FP4_VALUES, dtype=dtype, device=device)
        _FP4_LUT_CACHE[key] = cached
    return cached


def unpack_mxfp4_blocks(blocks: torch.Tensor, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    blocks = blocks.to(torch.uint8)
    lut = _fp4_lut(dtype, blocks.device)
    unpacked = torch.empty(*blocks.shape[:-1], blocks.shape[-1] * 2, dtype=dtype, device=blocks.device)
    unpacked[..., 0::2] = lut[(blocks & 0x0F).to(torch.long)]
    unpacked[..., 1::2] = lut[(blocks >> 4).to(torch.long)]
    return unpacked


def dequantize_mxfp4_projection(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    import math

    blocks = blocks.to(torch.uint8)
    scales = scales.to(torch.int32) - 127
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(f"MXFP4 blocks/scales shape mismatch: {blocks.shape[:-1]} != {scales.shape}")

    *prefix_shape, groups, packed_width = blocks.shape
    rows_total = math.prod(prefix_shape) * groups
    flat_blocks = blocks.reshape(rows_total, packed_width)
    flat_scales = scales.reshape(rows_total, 1)
    flat_out = torch.empty(rows_total, packed_width * 2, dtype=dtype, device=blocks.device)

    lut = _fp4_lut(dtype, blocks.device)
    for row_start in range(0, rows_total, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows_total)
        block_chunk = flat_blocks[row_start:row_end]
        scale_chunk = flat_scales[row_start:row_end]
        out_chunk = flat_out[row_start:row_end]
        out_chunk[:, 0::2] = lut[(block_chunk & 0x0F).to(torch.long)]
        out_chunk[:, 1::2] = lut[(block_chunk >> 4).to(torch.long)]
        torch.ldexp(out_chunk, scale_chunk, out=out_chunk)

    dense = flat_out.reshape(*prefix_shape, groups, packed_width * 2).view(*prefix_shape, groups * packed_width * 2)
    if dense.ndim < 3:
        dense = dense.unsqueeze(0).transpose(1, 2).contiguous().squeeze(0)
    else:
        dense = dense.transpose(1, 2).contiguous()
    return dense


def dequantize_mxfp4_expert(
    expert_state: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device | str] = None,
) -> Dict[str, torch.Tensor]:
    target_device = device or expert_state["gate_up_proj_blocks"].device
    gate_up = dequantize_mxfp4_projection(
        expert_state["gate_up_proj_blocks"].to(target_device),
        expert_state["gate_up_proj_scales"].to(target_device),
        dtype=dtype,
    )
    down = dequantize_mxfp4_projection(
        expert_state["down_proj_blocks"].to(target_device),
        expert_state["down_proj_scales"].to(target_device),
        dtype=dtype,
    )
    return {
        "gate_up_proj": gate_up,
        "gate_up_proj_bias": expert_state["gate_up_proj_bias"].to(device=target_device, dtype=dtype),
        "down_proj": down,
        "down_proj_bias": expert_state["down_proj_bias"].to(device=target_device, dtype=dtype),
    }


def _resolve_device(device: Optional[torch.device | str], fallback: torch.device | str) -> torch.device:
    resolved = torch.device(device or fallback)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested for MXFP4 execution ({resolved}), but torch.cuda.is_available() is false.")
    return resolved


def _sync_if_cuda(device: torch.device, *, enabled: bool = True) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_memory_snapshot(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "cuda_memory_allocated_bytes": 0,
            "cuda_memory_reserved_bytes": 0,
            "cuda_max_memory_allocated_bytes": 0,
            "cuda_max_memory_reserved_bytes": 0,
        }
    return {
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _check_cuda_budget(device: torch.device, max_vram_bytes: Optional[int], stage: str) -> None:
    if device.type != "cuda" or max_vram_bytes is None:
        return
    snapshot = _cuda_memory_snapshot(device)
    used = max(snapshot["cuda_memory_allocated_bytes"], snapshot["cuda_memory_reserved_bytes"])
    if used > int(max_vram_bytes):
        raise MemoryError(
            f"CUDA VRAM budget exceeded during {stage}: "
            f"allocated={snapshot['cuda_memory_allocated_bytes'] / 1024 ** 2:.1f}MB "
            f"reserved={snapshot['cuda_memory_reserved_bytes'] / 1024 ** 2:.1f}MB "
            f"budget={int(max_vram_bytes) / 1024 ** 2:.1f}MB"
        )


def _load_hf_gpt_oss_triton_kernel():
    global _HF_GPT_OSS_TRITON_KERNEL
    if _HF_GPT_OSS_TRITON_KERNEL is not None:
        return _HF_GPT_OSS_TRITON_KERNEL
    from transformers.integrations.hub_kernels import get_kernel
    import transformers.integrations.mxfp4 as hf_mxfp4

    hf_mxfp4.triton_kernels_hub = get_kernel("kernels-community/gpt-oss-triton-kernels")
    _HF_GPT_OSS_TRITON_KERNEL = hf_mxfp4
    return hf_mxfp4


def build_gpt_oss_hf_triton_selected_expert_module(
    expert_state: Mapping[str, torch.Tensor],
    *,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    device: Optional[torch.device | str] = None,
):
    target_device = torch.device(device or expert_state["gate_up_proj_blocks"].device)
    if target_device.type != "cuda":
        raise RuntimeError("HF GPT-OSS Triton MXFP4 execution requires CUDA")

    hf_mxfp4 = _load_hf_gpt_oss_triton_kernel()
    if hidden_size is None:
        hidden_size = int(expert_state["gate_up_proj_blocks"].shape[1]) * 32
    if intermediate_size is None:
        intermediate_size = int(expert_state["gate_up_proj_blocks"].shape[0]) // 2

    config = types.SimpleNamespace(
        num_local_experts=1,
        intermediate_size=int(intermediate_size),
        hidden_size=int(hidden_size),
        swiglu_limit=7.0,
    )
    module = hf_mxfp4.Mxfp4GptOssExperts(config).to(target_device)
    with torch.no_grad():
        module.gate_up_proj_bias.copy_(expert_state["gate_up_proj_bias"].to(device=target_device, dtype=torch.float32).unsqueeze(0))
        module.down_proj_bias.copy_(expert_state["down_proj_bias"].to(device=target_device, dtype=torch.float32).unsqueeze(0))

    hf_mxfp4.swizzle_mxfp4_convertops(
        expert_state["gate_up_proj_blocks"].unsqueeze(0).to(target_device),
        expert_state["gate_up_proj_scales"].unsqueeze(0).to(target_device),
        module,
        "gate_up_proj",
        target_device,
        hf_mxfp4.triton_kernels_hub,
    )
    hf_mxfp4.swizzle_mxfp4_convertops(
        expert_state["down_proj_blocks"].unsqueeze(0).to(target_device),
        expert_state["down_proj_scales"].unsqueeze(0).to(target_device),
        module,
        "down_proj",
        target_device,
        hf_mxfp4.triton_kernels_hub,
    )
    module._betterairllm_hf_mxfp4 = hf_mxfp4
    return module


def _module_tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return tensor_nbytes(value)
    storage = getattr(value, "storage", None)
    data = getattr(storage, "data", None)
    if isinstance(data, torch.Tensor):
        return tensor_nbytes(data)
    return 0


def gpt_oss_hf_triton_module_nbytes(module) -> int:
    total = 0
    for name in ("gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias"):
        total += _module_tensor_nbytes(getattr(module, name, None))
    for name in ("gate_up_proj_precision_config", "down_proj_precision_config"):
        precision_config = getattr(module, name, None)
        total += _module_tensor_nbytes(getattr(precision_config, "weight_scale", None))
    return total


def run_gpt_oss_selected_expert_hf_triton_timed(
    module,
    expert_input: torch.Tensor,
    *,
    output_device: Optional[torch.device | str] = None,
    sync_cuda_timing: bool = False,
) -> tuple[torch.Tensor, Dict[str, float]]:
    execution_device = expert_input.device
    if execution_device.type != "cuda":
        raise RuntimeError("HF GPT-OSS Triton MXFP4 execution requires CUDA inputs")
    hf_mxfp4 = getattr(module, "_betterairllm_hf_mxfp4", None)
    if hf_mxfp4 is None:
        raise RuntimeError("HF GPT-OSS Triton module is missing its kernel hub attachment")

    routing = hf_mxfp4.triton_kernels_hub.routing
    token_count = int(expert_input.shape[0])
    source_indices = torch.arange(token_count, device=execution_device, dtype=torch.int32)
    hist = torch.tensor([token_count], device=execution_device, dtype=torch.int32)
    gate_scal = torch.ones(token_count, device=execution_device, dtype=torch.float32)
    routing_data = routing.RoutingData(
        gate_scal,
        hist,
        1,
        1,
        routing.compute_expt_data_torch(hist, 1, token_count),
    )
    gather_idx = routing.GatherIndx(src_indx=source_indices, dst_indx=source_indices)
    scatter_idx = routing.ScatterIndx(src_indx=source_indices, dst_indx=source_indices)

    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    started = time.perf_counter()
    output = module(expert_input, routing_data, gather_idx, scatter_idx)
    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    output_device = torch.device(output_device or expert_input.device)
    return output.to(device=output_device, dtype=expert_input.dtype), {
        "hf_triton_kernel_seconds": time.perf_counter() - started,
        "mxfp4_dequant_seconds": 0.0,
        "routing_rows": token_count,
    }


def load_gpt_oss_expert_mxfp4_shard(layer_state_dict: Mapping[str, torch.Tensor], expert_id: int) -> Dict[str, torch.Tensor]:
    suffix_map = {
        "gate_up_proj_blocks": "gate_up_proj_blocks",
        "gate_up_proj_scales": "gate_up_proj_scales",
        "gate_up_proj_bias": "gate_up_proj_bias",
        "down_proj_blocks": "down_proj_blocks",
        "down_proj_scales": "down_proj_scales",
        "down_proj_bias": "down_proj_bias",
    }
    expert_state: Dict[str, torch.Tensor] = {}
    for key, tensor in layer_state_dict.items():
        for suffix, mapped_name in suffix_map.items():
            if key.endswith(suffix) or key.endswith(f"{suffix}.weight"):
                expert_state[mapped_name] = tensor.select(0, int(expert_id)).contiguous()
                break

    missing = set(suffix_map.values()) - set(expert_state)
    if missing:
        raise KeyError(f"GPT-OSS MXFP4 expert shard missing tensors: {sorted(missing)}")
    return expert_state


def run_gpt_oss_selected_expert_reference(
    expert_state: Mapping[str, torch.Tensor],
    expert_input: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    alpha: float = 1.702,
    limit: float = 7.0,
    mxfp4_device: Optional[torch.device | str] = None,
    matmul_device: Optional[torch.device | str] = None,
    output_device: Optional[torch.device | str] = None,
    max_vram_bytes: Optional[int] = None,
    sync_cuda_timing: bool = False,
) -> torch.Tensor:
    output, _ = run_gpt_oss_selected_expert_reference_timed(
        expert_state,
        expert_input,
        compute_dtype=compute_dtype,
        alpha=alpha,
        limit=limit,
        mxfp4_device=mxfp4_device,
        matmul_device=matmul_device,
        output_device=output_device,
        max_vram_bytes=max_vram_bytes,
        sync_cuda_timing=sync_cuda_timing,
    )
    return output


def run_gpt_oss_selected_expert_reference_timed(
    expert_state: Mapping[str, torch.Tensor],
    expert_input: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    alpha: float = 1.702,
    limit: float = 7.0,
    mxfp4_device: Optional[torch.device | str] = None,
    matmul_device: Optional[torch.device | str] = None,
    output_device: Optional[torch.device | str] = None,
    max_vram_bytes: Optional[int] = None,
    sync_cuda_timing: bool = False,
) -> tuple[torch.Tensor, Dict[str, float]]:
    timings: Dict[str, float] = {}
    execution_device = _resolve_device(matmul_device or mxfp4_device, expert_input.device)
    dequant_device = _resolve_device(mxfp4_device, execution_device)
    if dequant_device != execution_device:
        raise ValueError("The reference MXFP4 path requires mxfp4_device and expert_matmul_device to match.")
    output_device = torch.device(output_device or expert_input.device)
    device_report = {
        "hidden_states_device": str(expert_input.device),
        "expert_input_device": str(expert_input.device),
        "packed_gate_up_blocks_device": str(expert_state["gate_up_proj_blocks"].device),
        "packed_down_blocks_device": str(expert_state["down_proj_blocks"].device),
        "compute_dtype": str(compute_dtype),
        "requested_mxfp4_device": str(mxfp4_device) if mxfp4_device is not None else None,
        "requested_matmul_device": str(matmul_device) if matmul_device is not None else None,
        "execution_device": str(execution_device),
        "output_device": str(output_device),
    }
    _check_cuda_budget(execution_device, max_vram_bytes, "before selected expert dequant")
    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    started = time.perf_counter()
    dense = dequantize_mxfp4_expert(expert_state, dtype=compute_dtype, device=execution_device)
    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    timings["mxfp4_dequant_seconds"] = time.perf_counter() - started
    _check_cuda_budget(execution_device, max_vram_bytes, "after selected expert dequant")
    device_report.update({
        "dequantized_gate_up_device": str(dense["gate_up_proj"].device),
        "dequantized_down_device": str(dense["down_proj"].device),
        "dequantized_gate_up_dtype": str(dense["gate_up_proj"].dtype),
        "dequantized_down_dtype": str(dense["down_proj"].dtype),
        "matmul_device": str(execution_device),
    })

    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    started = time.perf_counter()
    current_state = expert_input.to(device=execution_device, dtype=compute_dtype)
    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    timings["input_cast_seconds"] = time.perf_counter() - started
    _check_cuda_budget(execution_device, max_vram_bytes, "after selected expert input cast")

    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    started = time.perf_counter()
    gate_up = current_state @ dense["gate_up_proj"] + dense["gate_up_proj_bias"]
    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    timings["gate_up_matmul_seconds"] = time.perf_counter() - started
    _check_cuda_budget(execution_device, max_vram_bytes, "after selected expert gate/up matmul")

    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    started = time.perf_counter()
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    timings["activation_seconds"] = time.perf_counter() - started
    _check_cuda_budget(execution_device, max_vram_bytes, "after selected expert activation")

    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    started = time.perf_counter()
    output = gated_output @ dense["down_proj"] + dense["down_proj_bias"]
    output = output.to(device=output_device, dtype=expert_input.dtype)
    _sync_if_cuda(execution_device, enabled=sync_cuda_timing)
    timings["down_matmul_seconds"] = time.perf_counter() - started
    timings["expert_matmul_seconds"] = timings["gate_up_matmul_seconds"] + timings["down_matmul_seconds"]
    device_report.update(_cuda_memory_snapshot(execution_device))
    timings["device_report"] = device_report
    del dense, current_state, gate_up, gate, up, glu, gated_output
    _check_cuda_budget(execution_device, max_vram_bytes, "after selected expert cleanup")
    return output, timings


def pack_dense_to_mxfp4_exact(dense: torch.Tensor, *, scale_exponent: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if dense.ndim != 3:
        raise ValueError(f"Expected dense projection [experts, in_dim, out_dim], got {tuple(dense.shape)}")
    if dense.shape[1] % 32 != 0:
        raise ValueError("MXFP4 test packing requires in_dim to be divisible by 32")
    storage = dense.transpose(1, 2).contiguous()
    experts, out_dim, in_dim = storage.shape
    grouped = storage.reshape(experts, out_dim, in_dim // 32, 32)
    scaled = grouped / float(2 ** scale_exponent)
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=dense.device)
    distances = (scaled.to(torch.float32).unsqueeze(-1) - lut).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)
    low = indices[..., 0::2]
    high = indices[..., 1::2]
    blocks = low | (high << 4)
    scales = torch.full(blocks.shape[:-1], 127 + int(scale_exponent), dtype=torch.uint8, device=dense.device)
    return blocks.contiguous(), scales.contiguous()


def mxfp4_state_nbytes(expert_state: Mapping[str, torch.Tensor]) -> int:
    return sum(tensor_nbytes(tensor) for tensor in expert_state.values())


def estimate_mxfp4_dequantized_expert_bytes(
    expert_state: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    gate_blocks = expert_state["gate_up_proj_blocks"]
    down_blocks = expert_state["down_proj_blocks"]
    gate_up_elems = int(gate_blocks.shape[0]) * int(gate_blocks.shape[1]) * int(gate_blocks.shape[2]) * 2
    down_elems = int(down_blocks.shape[0]) * int(down_blocks.shape[1]) * int(down_blocks.shape[2]) * 2
    bias_elems = int(expert_state["gate_up_proj_bias"].numel()) + int(expert_state["down_proj_bias"].numel())
    return (gate_up_elems + down_elems + bias_elems) * dtype_bytes
