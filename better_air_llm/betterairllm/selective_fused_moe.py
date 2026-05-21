from __future__ import annotations

from collections import OrderedDict
import time
from typing import Callable, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

from .gpt_oss_mxfp4 import (
    build_gpt_oss_hf_triton_selected_expert_module,
    estimate_mxfp4_dequantized_expert_bytes,
    gpt_oss_hf_triton_module_nbytes,
    run_gpt_oss_selected_expert_hf_triton_timed,
    run_gpt_oss_selected_expert_reference_timed,
)


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def state_dict_nbytes(state_dict):
    return sum(tensor_nbytes(tensor) for tensor in state_dict.values())


class SelectiveFusedMoEAdapter:

    adapter_name = "selective_fused_moe_base"

    def __init__(
        self,
        router,
        expert_loader: Callable[[int], Mapping[str, torch.Tensor]],
        num_experts: int,
        top_k: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        normalize_router_weights: bool = True,
        cache_size: int = 0,
        batch_expert_loader: Optional[Callable[[Iterable[int]], Mapping[int, Mapping[str, torch.Tensor]]]] = None,
        expert_execution_mode: str = "grouped_by_expert",
    ) -> None:
        self.router = router
        self.expert_loader = expert_loader
        self.batch_expert_loader = batch_expert_loader
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.activation = activation
        self.normalize_router_weights = normalize_router_weights
        self.cache_size = max(0, int(cache_size))
        if expert_execution_mode not in {"per_token", "grouped_by_expert"}:
            raise ValueError(f"unsupported expert_execution_mode: {expert_execution_mode}")
        self.expert_execution_mode = expert_execution_mode
        self._expert_cache: OrderedDict[int, Mapping[str, torch.Tensor]] = OrderedDict()
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {
            "expert_load_calls": 0,
            "expert_load_bytes": 0,
            "expert_cache_hits": 0,
            "expert_cache_misses": 0,
            "selected_expert_slots": 0,
            "unique_selected_experts": 0,
            "packed_expert_bytes_loaded": 0,
            "dequantized_temporary_bytes": 0,
            "mxfp4_dequant_seconds": 0.0,
            "expert_matmul_seconds": 0.0,
            "gate_up_matmul_seconds": 0.0,
            "down_matmul_seconds": 0.0,
            "activation_seconds": 0.0,
            "input_cast_seconds": 0.0,
            "device_report": {},
            "layer_timing_breakdown": {},
            "expert_execution_mode": self.expert_execution_mode,
            "run_expert_calls": 0,
            "grouped_expert_calls": 0,
            "per_token_expert_calls": 0,
            "expert_input_rows": 0,
            "mxfp4_execution": getattr(self, "mxfp4_execution", "reference_cuda"),
            "hf_triton_kernel_seconds": 0.0,
            "hf_triton_module_cache_hits": 0,
            "hf_triton_module_cache_misses": 0,
            "hf_triton_module_builds": 0,
            "hf_triton_module_build_seconds": 0.0,
            "hf_triton_module_cache_bytes": int(getattr(self, "_hf_triton_module_cache_state", {}).get("bytes", 0)),
            "hf_triton_module_cache_budget_bytes": getattr(self, "hf_triton_module_cache_budget_bytes", 0),
            "hf_triton_module_cache_min_requests": getattr(self, "hf_triton_module_cache_min_requests", 1),
            "hf_triton_module_cache_evictions": 0,
            "hf_triton_module_admission_skips": 0,
            "hf_triton_module_unique_requests": len(getattr(self, "_hf_triton_module_request_counts", {})),
            "hf_triton_module_direct_loads_skipped": 0,
            "hf_triton_fallbacks": 0,
            "hf_triton_last_error": "",
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_dim = original_shape[-1]
        flat_states = hidden_states.reshape(-1, hidden_dim)

        router_logits = self.router(flat_states)
        router_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.normalize_router_weights:
            router_weights = torch.softmax(router_weights, dim=-1)

        output = torch.zeros_like(flat_states)
        unique_experts = torch.unique(selected_experts).tolist()
        self.stats["selected_expert_slots"] += int(selected_experts.numel())
        self.stats["unique_selected_experts"] += len(unique_experts)
        expert_states = self._get_experts(int(expert_id) for expert_id in unique_experts)

        if self.expert_execution_mode == "per_token":
            for token_position in range(selected_experts.shape[0]):
                for expert_slot in range(selected_experts.shape[1]):
                    expert_id = int(selected_experts[token_position, expert_slot])
                    expert_state = expert_states[expert_id]
                    expert_input = flat_states[token_position:token_position + 1]
                    self.stats["per_token_expert_calls"] += 1
                    self.stats["run_expert_calls"] += 1
                    self.stats["expert_input_rows"] += int(expert_input.shape[0])
                    expert_output = self._run_expert(expert_state, expert_input)
                    expert_output = expert_output * router_weights[token_position, expert_slot].view(1, 1)
                    output.index_add_(0, selected_experts.new_tensor([token_position]), expert_output)
            return output.reshape(original_shape)

        for expert_id in unique_experts:
            expert_id = int(expert_id)
            expert_state = expert_states[expert_id]

            token_positions, expert_slots = torch.where(selected_experts == expert_id)
            expert_input = flat_states[token_positions]
            self.stats["grouped_expert_calls"] += 1
            self.stats["run_expert_calls"] += 1
            self.stats["expert_input_rows"] += int(expert_input.shape[0])
            expert_output = self._run_expert(expert_state, expert_input)
            expert_output = expert_output * router_weights[token_positions, expert_slots].unsqueeze(-1)
            output.index_add_(0, token_positions, expert_output)

        return output.reshape(original_shape)

    def _get_expert(self, expert_id: int) -> Mapping[str, torch.Tensor]:
        return self._get_experts([expert_id])[int(expert_id)]

    def _get_experts(self, expert_ids: Iterable[int]) -> Dict[int, Mapping[str, torch.Tensor]]:
        result: Dict[int, Mapping[str, torch.Tensor]] = {}
        missing = []
        for expert_id in [int(item) for item in expert_ids]:
            if expert_id in result:
                continue
            if expert_id in self._expert_cache:
                expert_state = self._expert_cache.pop(expert_id)
                self._expert_cache[expert_id] = expert_state
                self.stats["expert_cache_hits"] += 1
                result[expert_id] = expert_state
            else:
                missing.append(expert_id)

        if missing:
            if self.batch_expert_loader is not None:
                loaded = {int(key): value for key, value in self.batch_expert_loader(missing).items()}
            else:
                loaded = {expert_id: self.expert_loader(expert_id) for expert_id in missing}

            for expert_id in missing:
                expert_state = loaded[expert_id]
                result[expert_id] = expert_state
                self.stats["expert_cache_misses"] += 1
                self.stats["expert_load_calls"] += 1
                self.stats["expert_load_bytes"] += state_dict_nbytes(expert_state)
                if self.cache_size > 0:
                    self._expert_cache[expert_id] = expert_state
                    while len(self._expert_cache) > self.cache_size:
                        self._expert_cache.popitem(last=False)

        return result

    def _run_expert(self, expert_state: Mapping[str, torch.Tensor], expert_input: torch.Tensor) -> torch.Tensor:
        gate_up_weight = expert_state["gate_up_proj.weight"].to(device=expert_input.device, dtype=expert_input.dtype)
        down_weight = expert_state["down_proj.weight"].to(device=expert_input.device, dtype=expert_input.dtype)

        gate_up = F.linear(expert_input, gate_up_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self.activation(gate) * up
        return F.linear(activated, down_weight)


class FakeFusedMoEAdapter(SelectiveFusedMoEAdapter):
    adapter_name = "fake_fused_moe"


class Qwen35SelectiveFusedMoEAdapter(SelectiveFusedMoEAdapter):

    adapter_name = "qwen3_5_moe"

    def __init__(self, *args, layer_name: Optional[str] = None, **kwargs) -> None:
        self.layer_name = layer_name
        super().__init__(*args, normalize_router_weights=False, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        unique_experts = torch.unique(top_k_index).tolist()
        self.stats["selected_expert_slots"] += int(top_k_index.numel())
        self.stats["unique_selected_experts"] += len(unique_experts)
        expert_states = self._get_experts(int(expert_id) for expert_id in unique_experts)

        if self.expert_execution_mode == "per_token":
            for token_position in range(top_k_index.shape[0]):
                for top_k_pos in range(top_k_index.shape[1]):
                    expert_id = int(top_k_index[token_position, top_k_pos])
                    expert_input = hidden_states[token_position:token_position + 1]
                    if self.layer_name is not None:
                        self._record_selected_expert(expert_id)
                    self.stats["per_token_expert_calls"] += 1
                    self.stats["run_expert_calls"] += 1
                    self.stats["expert_input_rows"] += int(expert_input.shape[0])
                    expert_output = self._run_expert(expert_states[expert_id], expert_input)
                    expert_output = expert_output * top_k_weights[token_position, top_k_pos].view(1, 1)
                    output.index_add_(0, top_k_index.new_tensor([token_position]), expert_output.to(output.dtype))
            return output

        for expert_id in unique_experts:
            expert_id = int(expert_id)
            token_idx, top_k_pos = torch.where(top_k_index == expert_id)
            expert_input = hidden_states[token_idx]
            if self.layer_name is not None:
                self._record_selected_expert(expert_id)
            self.stats["grouped_expert_calls"] += 1
            self.stats["run_expert_calls"] += 1
            self.stats["expert_input_rows"] += int(expert_input.shape[0])
            expert_output = self._run_expert(expert_states[expert_id], expert_input)
            expert_output = expert_output * top_k_weights[token_idx, top_k_pos, None]
            output.index_add_(0, token_idx, expert_output.to(output.dtype))

        return output

    def _record_selected_expert(self, expert_id: int) -> None:
        layer_stats = self.stats["layer_timing_breakdown"].setdefault(
            self.layer_name,
            {
                "selected_experts": [],
                "selected_expert_count": 0,
            },
        )
        if expert_id not in layer_stats["selected_experts"]:
            layer_stats["selected_experts"].append(expert_id)
            layer_stats["selected_expert_count"] = len(layer_stats["selected_experts"])


class GptOssSelectiveFusedMoEAdapter(SelectiveFusedMoEAdapter):

    adapter_name = "gpt_oss"

    def __init__(self, *args, alpha: float = 1.702, limit: float = 7.0, **kwargs) -> None:
        self.layer_name = kwargs.pop("layer_name", None)
        self.mxfp4_device = kwargs.pop("mxfp4_device", None)
        self.expert_matmul_device = kwargs.pop("expert_matmul_device", None)
        self.mxfp4_execution = kwargs.pop("mxfp4_execution", "reference_cuda")
        if self.mxfp4_execution == "triton_fused":
            self.mxfp4_execution = "hf_triton"
        if self.mxfp4_execution not in {"reference_cuda", "hf_triton"}:
            raise ValueError(f"unsupported mxfp4_execution: {self.mxfp4_execution}")
        self.hf_triton_module_cache_budget_bytes = max(0, int(kwargs.pop("hf_triton_module_cache_mb", 0))) * 1024 * 1024
        self.hf_triton_module_cache_min_requests = max(1, int(kwargs.pop("hf_triton_module_cache_min_requests", 2)))
        self._hf_triton_module_cache: OrderedDict[object, tuple[object, int, Mapping[str, torch.Tensor]]] = kwargs.pop(
            "hf_triton_shared_module_cache",
            OrderedDict(),
        )
        self._hf_triton_module_cache_state = kwargs.pop(
            "hf_triton_shared_module_cache_state",
            {"bytes": 0, "budget_bytes": self.hf_triton_module_cache_budget_bytes},
        )
        self._hf_triton_module_request_counts = kwargs.pop("hf_triton_shared_module_request_counts", {})
        self.max_vram_bytes = kwargs.pop("max_vram_bytes", None)
        self.sync_cuda_timing = bool(kwargs.pop("sync_cuda_timing", False))
        super().__init__(*args, normalize_router_weights=False, **kwargs)
        self.alpha = alpha
        self.limit = limit

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_dim = original_shape[-1]
        flat_states = hidden_states.reshape(-1, hidden_dim)

        if router_indices is None or routing_weights is None:
            _, routing_weights, router_indices = self.router(flat_states)

        output = torch.zeros_like(flat_states)
        unique_experts = torch.unique(router_indices).tolist()
        self.stats["selected_expert_slots"] += int(router_indices.numel())
        self.stats["unique_selected_experts"] += len(unique_experts)
        if self.mxfp4_execution == "hf_triton":
            cached_modules = {}
            missing_experts = []
            for expert_id in [int(expert_id) for expert_id in unique_experts]:
                module = self._get_cached_hf_triton_module(expert_id)
                if module is None:
                    missing_experts.append(expert_id)
                else:
                    cached_modules[expert_id] = module
                    self.stats["hf_triton_module_direct_loads_skipped"] += 1
            expert_states = self._get_experts(missing_experts) if missing_experts else {}
        else:
            cached_modules = {}
            expert_states = self._get_experts(int(expert_id) for expert_id in unique_experts)

        if self.expert_execution_mode == "per_token":
            for token_position in range(router_indices.shape[0]):
                for top_k_pos in range(router_indices.shape[1]):
                    expert_id = int(router_indices[token_position, top_k_pos])
                    expert_input = flat_states[token_position:token_position + 1]
                    if self.layer_name is not None:
                        self._record_selected_expert(self.layer_name, expert_id)
                    self.stats["per_token_expert_calls"] += 1
                    self.stats["run_expert_calls"] += 1
                    self.stats["expert_input_rows"] += int(expert_input.shape[0])
                    if expert_id in cached_modules:
                        expert_output = self._run_cached_hf_triton_module(cached_modules[expert_id], expert_input)
                    else:
                        expert_output = self._run_expert(expert_states[expert_id], expert_input, expert_id=expert_id)
                    expert_output = expert_output * routing_weights[token_position, top_k_pos].view(1, 1)
                    output.index_add_(0, router_indices.new_tensor([token_position]), expert_output.to(flat_states.dtype))
            return output.reshape(original_shape)

        for expert_id in unique_experts:
            expert_id = int(expert_id)
            token_idx, top_k_pos = torch.where(router_indices == expert_id)
            expert_input = flat_states[token_idx]
            if self.layer_name is not None:
                self._record_selected_expert(self.layer_name, expert_id)
            self.stats["grouped_expert_calls"] += 1
            self.stats["run_expert_calls"] += 1
            self.stats["expert_input_rows"] += int(expert_input.shape[0])
            if expert_id in cached_modules:
                expert_output = self._run_cached_hf_triton_module(cached_modules[expert_id], expert_input)
            else:
                expert_output = self._run_expert(expert_states[expert_id], expert_input, expert_id=expert_id)
            expert_output = expert_output * routing_weights[token_idx, top_k_pos, None]
            output.index_add_(0, token_idx, expert_output.to(flat_states.dtype))

        return output.reshape(original_shape)

    def _record_selected_expert(self, layer_name: str, expert_id: int) -> None:
        layer_stats = self.stats["layer_timing_breakdown"].setdefault(
            layer_name,
            {
                "selected_experts": [],
                "selected_expert_count": 0,
                "mxfp4_dequant_seconds": 0.0,
                "expert_matmul_seconds": 0.0,
                "gate_up_matmul_seconds": 0.0,
                "down_matmul_seconds": 0.0,
                "activation_seconds": 0.0,
                "input_cast_seconds": 0.0,
            },
        )
        if expert_id not in layer_stats["selected_experts"]:
            layer_stats["selected_experts"].append(expert_id)
            layer_stats["selected_expert_count"] = len(layer_stats["selected_experts"])

    def _hf_triton_cache_key(
        self,
        expert_state: Mapping[str, torch.Tensor],
        expert_input: torch.Tensor,
        expert_id: Optional[int] = None,
    ) -> object:
        if expert_id is not None:
            return (self.layer_name, int(expert_id))
        return id(expert_state)

    def _get_hf_triton_module(
        self,
        expert_state: Mapping[str, torch.Tensor],
        expert_input: torch.Tensor,
        expert_id: Optional[int] = None,
    ):
        key = self._hf_triton_cache_key(expert_state, expert_input, expert_id)
        if key in self._hf_triton_module_cache:
            module, module_bytes, cached_expert_state = self._hf_triton_module_cache.pop(key)
            self._hf_triton_module_cache[key] = (module, module_bytes, cached_expert_state)
            self.stats["hf_triton_module_cache_hits"] += 1
            return module

        self.stats["hf_triton_module_cache_misses"] += 1
        request_count = int(self._hf_triton_module_request_counts.get(key, 0)) + 1
        self._hf_triton_module_request_counts[key] = request_count
        self.stats["hf_triton_module_unique_requests"] = len(self._hf_triton_module_request_counts)
        if request_count < self.hf_triton_module_cache_min_requests:
            self.stats["hf_triton_module_admission_skips"] += 1
            return None
        device = torch.device(self.expert_matmul_device or self.mxfp4_device or expert_input.device)
        if device.type != "cuda":
            raise RuntimeError("HF Triton selected-expert execution requires CUDA")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        module = build_gpt_oss_hf_triton_selected_expert_module(
            expert_state,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.stats["hf_triton_module_builds"] += 1
        self.stats["hf_triton_module_build_seconds"] += time.perf_counter() - started

        module_bytes = max(gpt_oss_hf_triton_module_nbytes(module), state_dict_nbytes(expert_state))
        if self.max_vram_bytes is not None and device.type == "cuda":
            used = max(torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device))
            if used > int(self.max_vram_bytes):
                raise MemoryError(
                    "CUDA VRAM budget exceeded during HF Triton selected-expert module build: "
                    f"allocated={torch.cuda.memory_allocated(device) / 1024 ** 2:.1f}MB "
                    f"reserved={torch.cuda.memory_reserved(device) / 1024 ** 2:.1f}MB "
                    f"budget={int(self.max_vram_bytes) / 1024 ** 2:.1f}MB"
                )
        if self.hf_triton_module_cache_budget_bytes > 0 and module_bytes <= self.hf_triton_module_cache_budget_bytes:
            self._hf_triton_module_cache[key] = (module, module_bytes, expert_state)
            self._hf_triton_module_cache_state["bytes"] = int(self._hf_triton_module_cache_state.get("bytes", 0)) + module_bytes
            self._enforce_hf_triton_module_cache_budget()
        self.stats["hf_triton_module_cache_bytes"] = int(self._hf_triton_module_cache_state.get("bytes", 0))
        return module

    def _get_cached_hf_triton_module(self, expert_id: int):
        key = (self.layer_name, int(expert_id))
        if key not in self._hf_triton_module_cache:
            return None
        module, module_bytes, cached_expert_state = self._hf_triton_module_cache.pop(key)
        self._hf_triton_module_cache[key] = (module, module_bytes, cached_expert_state)
        self.stats["hf_triton_module_cache_hits"] += 1
        return module

    def _enforce_hf_triton_module_cache_budget(self) -> None:
        while (
            int(self._hf_triton_module_cache_state.get("bytes", 0)) > self.hf_triton_module_cache_budget_bytes
            and self._hf_triton_module_cache
        ):
            _, (_, module_bytes, _) = self._hf_triton_module_cache.popitem(last=False)
            self.stats["hf_triton_module_cache_evictions"] += 1
            self._hf_triton_module_cache_state["bytes"] = max(
                0,
                int(self._hf_triton_module_cache_state.get("bytes", 0)) - module_bytes,
            )

    def _run_cached_hf_triton_module(self, module, expert_input: torch.Tensor) -> torch.Tensor:
        output, timings = run_gpt_oss_selected_expert_hf_triton_timed(
            module,
            expert_input.to(device=torch.device(self.expert_matmul_device or self.mxfp4_device or expert_input.device)),
            output_device=expert_input.device,
            sync_cuda_timing=self.sync_cuda_timing,
        )
        self.stats["hf_triton_kernel_seconds"] += float(timings.get("hf_triton_kernel_seconds") or 0.0)
        if self.layer_name is not None:
            layer_stats = self.stats["layer_timing_breakdown"][self.layer_name]
            layer_stats["hf_triton_kernel_seconds"] = layer_stats.get("hf_triton_kernel_seconds", 0.0) + float(
                timings.get("hf_triton_kernel_seconds") or 0.0
            )
        return output

    def _run_packed_hf_triton_expert(
        self,
        expert_state: Mapping[str, torch.Tensor],
        expert_input: torch.Tensor,
        expert_id: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        module = self._get_hf_triton_module(expert_state, expert_input, expert_id)
        if module is None:
            return None
        output, timings = run_gpt_oss_selected_expert_hf_triton_timed(
            module,
            expert_input.to(device=torch.device(self.expert_matmul_device or self.mxfp4_device or expert_input.device)),
            output_device=expert_input.device,
            sync_cuda_timing=self.sync_cuda_timing,
        )
        self.stats["hf_triton_kernel_seconds"] += float(timings.get("hf_triton_kernel_seconds") or 0.0)
        device = torch.device(self.expert_matmul_device or self.mxfp4_device or expert_input.device)
        if self.max_vram_bytes is not None and device.type == "cuda":
            used = max(torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device))
            if used > int(self.max_vram_bytes):
                raise MemoryError(
                    "CUDA VRAM budget exceeded during HF Triton selected-expert forward: "
                    f"allocated={torch.cuda.memory_allocated(device) / 1024 ** 2:.1f}MB "
                    f"reserved={torch.cuda.memory_reserved(device) / 1024 ** 2:.1f}MB "
                    f"budget={int(self.max_vram_bytes) / 1024 ** 2:.1f}MB"
                )
        self.stats["mxfp4_dequant_seconds"] += float(timings.get("mxfp4_dequant_seconds") or 0.0)
        if self.layer_name is not None:
            layer_stats = self.stats["layer_timing_breakdown"][self.layer_name]
            layer_stats["mxfp4_dequant_seconds"] += float(timings.get("mxfp4_dequant_seconds") or 0.0)
            layer_stats["hf_triton_kernel_seconds"] = layer_stats.get("hf_triton_kernel_seconds", 0.0) + float(
                timings.get("hf_triton_kernel_seconds") or 0.0
            )
        return output

    def _run_expert(
        self,
        expert_state: Mapping[str, torch.Tensor],
        expert_input: torch.Tensor,
        expert_id: Optional[int] = None,
    ) -> torch.Tensor:
        if "gate_up_proj_blocks" in expert_state:
            compute_dtype = torch.bfloat16 if expert_input.dtype != torch.float16 else torch.float16
            self.stats["packed_expert_bytes_loaded"] += state_dict_nbytes(expert_state)
            self.stats["dequantized_temporary_bytes"] += estimate_mxfp4_dequantized_expert_bytes(
                expert_state,
                dtype=compute_dtype,
            )
            if self.mxfp4_execution == "hf_triton":
                try:
                    triton_output = self._run_packed_hf_triton_expert(expert_state, expert_input, expert_id)
                    if triton_output is not None:
                        return triton_output
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, MemoryError)):
                        raise
                    self.stats["hf_triton_fallbacks"] += 1
                    self.stats["hf_triton_last_error"] = f"{type(exc).__name__}: {exc}"
            output, timings = run_gpt_oss_selected_expert_reference_timed(
                expert_state,
                expert_input,
                compute_dtype=compute_dtype,
                alpha=self.alpha,
                limit=self.limit,
                mxfp4_device=self.mxfp4_device,
                matmul_device=self.expert_matmul_device,
                output_device=expert_input.device,
                max_vram_bytes=self.max_vram_bytes,
                sync_cuda_timing=self.sync_cuda_timing,
            )
            for key, value in timings.items():
                if isinstance(value, (int, float)):
                    self.stats[key] = self.stats.get(key, 0.0) + float(value)
                    if self.layer_name is not None:
                        self.stats["layer_timing_breakdown"][self.layer_name][key] += float(value)
                elif key == "device_report" and isinstance(value, dict):
                    self.stats["device_report"].update(value)
            return output

        gate_up_weight = expert_state["gate_up_proj"].to(device=expert_input.device, dtype=expert_input.dtype)
        gate_up_bias = expert_state["gate_up_proj_bias"].to(device=expert_input.device, dtype=expert_input.dtype)
        down_weight = expert_state["down_proj"].to(device=expert_input.device, dtype=expert_input.dtype)
        down_bias = expert_state["down_proj_bias"].to(device=expert_input.device, dtype=expert_input.dtype)

        gate_up = expert_input @ gate_up_weight + gate_up_bias
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        return gated_output @ down_weight + down_bias


def build_fake_fused_expert_shards(gate_up_proj_weight: torch.Tensor, down_proj_weight: torch.Tensor):
    shards: Dict[int, Dict[str, torch.Tensor]] = {}
    for expert_id in range(gate_up_proj_weight.shape[0]):
        shards[expert_id] = {
            "gate_up_proj.weight": gate_up_proj_weight[expert_id].contiguous(),
            "down_proj.weight": down_proj_weight[expert_id].contiguous(),
        }
    return shards


def build_fake_gpt_oss_expert_shards(
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
):
    shards: Dict[int, Dict[str, torch.Tensor]] = {}
    for expert_id in range(gate_up_proj.shape[0]):
        shards[expert_id] = {
            "gate_up_proj": gate_up_proj[expert_id].contiguous(),
            "gate_up_proj_bias": gate_up_proj_bias[expert_id].contiguous(),
            "down_proj": down_proj[expert_id].contiguous(),
            "down_proj_bias": down_proj_bias[expert_id].contiguous(),
        }
    return shards


def fake_full_fused_moe_forward(
    hidden_states: torch.Tensor,
    router,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    top_k: int,
    activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    normalize_router_weights: bool = True,
) -> torch.Tensor:
    original_shape = hidden_states.shape
    hidden_dim = original_shape[-1]
    flat_states = hidden_states.reshape(-1, hidden_dim)

    router_logits = router(flat_states)
    router_weights, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    if normalize_router_weights:
        router_weights = torch.softmax(router_weights, dim=-1)

    output = torch.zeros_like(flat_states)
    for expert_id in torch.unique(selected_experts).tolist():
        expert_id = int(expert_id)
        token_positions, expert_slots = torch.where(selected_experts == expert_id)
        expert_input = flat_states[token_positions]
        gate_up = F.linear(expert_input, gate_up_proj_weight[expert_id].to(device=expert_input.device, dtype=expert_input.dtype))
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(
            activation(gate) * up,
            down_proj_weight[expert_id].to(device=expert_input.device, dtype=expert_input.dtype),
        )
        expert_output = expert_output * router_weights[token_positions, expert_slots].unsqueeze(-1)
        output.index_add_(0, token_positions, expert_output)

    return output.reshape(original_shape)


def fake_full_expert_bytes(gate_up_proj_weight: torch.Tensor, down_proj_weight: torch.Tensor) -> int:
    return tensor_nbytes(gate_up_proj_weight) + tensor_nbytes(down_proj_weight)


def fake_gpt_oss_full_expert_bytes(
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
) -> int:
    return (
        tensor_nbytes(gate_up_proj)
        + tensor_nbytes(gate_up_proj_bias)
        + tensor_nbytes(down_proj)
        + tensor_nbytes(down_proj_bias)
    )


def fake_gpt_oss_full_fused_moe_forward(
    hidden_states: torch.Tensor,
    router,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    original_shape = hidden_states.shape
    hidden_dim = original_shape[-1]
    flat_states = hidden_states.reshape(-1, hidden_dim)
    _, routing_weights, router_indices = router(flat_states)

    output = torch.zeros_like(flat_states)
    for expert_id in torch.unique(router_indices).tolist():
        expert_id = int(expert_id)
        token_idx, top_k_pos = torch.where(router_indices == expert_id)
        current_state = flat_states[token_idx]
        gate_up = current_state @ gate_up_proj[expert_id].to(current_state.dtype)
        gate_up = gate_up + gate_up_proj_bias[expert_id].to(current_state.dtype)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        gated_output = (up + 1) * glu
        out = gated_output @ down_proj[expert_id].to(current_state.dtype)
        out = out + down_proj_bias[expert_id].to(current_state.dtype)
        weighted_output = out * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, weighted_output.to(flat_states.dtype))

    return output.reshape(original_shape)
