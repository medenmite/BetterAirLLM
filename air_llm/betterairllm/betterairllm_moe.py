from collections import OrderedDict
import json
from pathlib import Path
import time
import types

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device

from .airllm_base import BetterAirLLMBaseModel
from .gpt_oss_mxfp4 import dequantize_mxfp4_expert
from .selective_fused_moe import GptOssSelectiveFusedMoEAdapter, Qwen35SelectiveFusedMoEAdapter
from .utils import clean_memory


class BetterAirLLMMoE(BetterAirLLMBaseModel):
    """Generic MoE BetterAirLLM runtime.

    MoE checkpoints are split into a dense shard per transformer block plus
    one shard for each ``*.experts.<id>`` module. During layer execution the
    HuggingFace MoE implementation still owns routing, but each expert loads
    its own shard lazily when the router calls that expert.
    """

    def __init__(self, *args, **kwargs):
        self._qwen35_moe_layout = bool(kwargs.pop("qwen3_5_moe_layout", False))
        self.moe_expert_cache_mb = kwargs.pop("moe_expert_cache_mb", None)
        self.moe_cpu_expert_cache_mb = kwargs.pop("moe_cpu_expert_cache_mb", 0)
        self.vram_cache_mode = kwargs.pop("vram_cache_mode", "off")
        self.vram_cache_mb = kwargs.pop("vram_cache_mb", 0)
        self.dense_expert_cache_mb = kwargs.pop("dense_expert_cache_mb", None)
        self.dense_expert_cache_min_requests = int(kwargs.pop(
            "dense_expert_cache_min_requests",
            1 if self._qwen35_moe_layout else 2,
        ))
        self.dense_expert_cache_max_builds_without_hit = int(kwargs.pop("dense_expert_cache_max_builds_without_hit", 256))
        self.expert_execution_mode = kwargs.pop("expert_execution_mode", "grouped_by_expert")
        self.mxfp4_execution = kwargs.pop("mxfp4_execution", "reference_cuda")
        self.hf_triton_module_cache_mb = kwargs.pop("hf_triton_module_cache_mb", 0)
        self.hf_triton_module_cache_min_requests = int(kwargs.pop("hf_triton_module_cache_min_requests", 2))
        self._hf_triton_module_cache = OrderedDict()
        self._hf_triton_module_cache_state = {
            "bytes": 0,
            "budget_bytes": max(0, int(self.hf_triton_module_cache_mb)) * 1024 * 1024,
        }
        self._hf_triton_module_request_counts = {}
        self.enable_gpt_oss_batch_direct_slice = kwargs.pop("enable_gpt_oss_batch_direct_slice", False)
        self.enable_qwen35_batch_direct_slice = kwargs.pop("enable_qwen35_batch_direct_slice", True)
        self._resident_expert_cache = OrderedDict()
        self._resident_expert_bytes = 0
        self._cpu_expert_cache = OrderedDict()
        self._cpu_expert_bytes = 0
        self._gpu_packed_expert_cache = OrderedDict()
        self._gpu_packed_expert_bytes = 0
        self._gpu_packed_expert_cache_hits = 0
        self._gpu_packed_expert_cache_misses = 0
        self._gpu_packed_expert_bytes_avoided = 0
        self._gpu_dense_expert_cache = OrderedDict()
        self._gpu_dense_expert_bytes = 0
        self._gpu_dense_expert_cache_hits = 0
        self._gpu_dense_expert_cache_misses = 0
        self._gpu_dense_expert_bytes_avoided = 0
        self._gpu_dense_expert_builds = 0
        self._gpu_dense_expert_build_seconds = 0.0
        self._gpu_dense_expert_request_counts = {}
        self._gpu_dense_expert_admission_skips = 0
        self._gpu_dense_expert_disabled_no_hits = False
        self._expert_modules = {}
        self._expert_cache_hits = 0
        self._expert_cache_misses = 0
        self._cpu_expert_cache_hits = 0
        self._cpu_expert_cache_misses = 0
        self._timed_module_stats = {}
        self._moe_manifest = {}
        kwargs["moe_expert_sharding"] = True
        super(BetterAirLLMMoE, self).__init__(*args, **kwargs)
        self._resident_expert_budget_bytes = self._resolve_expert_cache_budget(self.moe_expert_cache_mb)
        self._cpu_expert_budget_bytes = max(0, int(self.moe_cpu_expert_cache_mb)) * 1024 * 1024
        self._gpu_packed_expert_budget_bytes = self._resolve_gpu_packed_expert_cache_budget()
        self._gpu_dense_expert_budget_bytes = self._resolve_gpu_dense_expert_cache_budget()

    def set_layer_names_dict(self):
        if getattr(self, "_qwen35_moe_layout", False):
            self.layer_names_dict = {
                "embed": "model.language_model.embed_tokens",
                "layer_prefix": "model.language_model.layers",
                "norm": "model.language_model.norm",
                "lm_head": "lm_head",
            }
            return
        super().set_layer_names_dict()

    def get_use_better_transformer(self):
        return False

    def init_model(self):
        super().init_model()
        self._load_moe_manifest()
        self._install_lazy_expert_loaders()

    def _install_lazy_expert_loaders(self):
        layer_prefix = self.layer_names_dict["layer_prefix"]

        for layer_idx, layer in enumerate(self.layers[1:-2]):
            layer_name = f"{layer_prefix}.{layer_idx}"
            for module_name, module in layer.named_modules():
                self._maybe_patch_timed_module(layer_name, module_name, module)
                if self._maybe_patch_gpt_oss_fused_experts(layer_name, module_name, module):
                    continue
                if self._maybe_patch_qwen35_fused_experts(layer_name, module_name, module):
                    continue
                for expert_key, expert in self._iter_experts(module_name, module):
                    expert_list_name = f"{layer_name}.{module_name}" if module_name else layer_name
                    shard_name = f"{expert_list_name}.{expert_key}"
                    self._expert_modules[shard_name] = expert
                    self._patch_expert_forward(expert, shard_name)

        if not self._expert_modules:
            fused_count = self._moe_manifest.get("fused_tensor_count", 0)
            if fused_count:
                print("WARNING: BetterAirLLMMoE found fused expert shards, but no ModuleList/ModuleDict experts. "
                      "Selective fused execution is disabled; dense fused tensors stay loaded for correctness.")
            else:
                print("WARNING: BetterAirLLMMoE did not find ModuleList/ModuleDict experts. "
                      "The model will fall back to dense layer streaming.")

    def _maybe_patch_gpt_oss_fused_experts(self, layer_name, module_name, module):
        if self._moe_manifest.get("adapter_name") not in {"gpt_oss", "gpt_oss_mxfp4_reference"}:
            return False
        if not self._is_gpt_oss_experts_module(module):
            return False

        shard_prefix = f"{layer_name}.{module_name}" if module_name else layer_name
        owner = self

        def expert_loader(expert_id):
            return owner._load_gpt_oss_expert_state_dict(f"{shard_prefix}.{int(expert_id)}")

        def batch_expert_loader(expert_ids):
            return owner._load_gpt_oss_expert_state_dicts(shard_prefix, expert_ids)

        adapter = GptOssSelectiveFusedMoEAdapter(
            router=None,
            expert_loader=expert_loader,
            batch_expert_loader=batch_expert_loader if self.enable_gpt_oss_batch_direct_slice else None,
            num_experts=module.num_experts,
            top_k=getattr(self.config, "num_experts_per_tok", 4),
            alpha=getattr(module, "alpha", 1.702),
            limit=getattr(module, "limit", 7.0),
            cache_size=0,
            expert_execution_mode=self.expert_execution_mode,
            layer_name=layer_name,
            mxfp4_device=self.mxfp4_device,
            expert_matmul_device=self.expert_matmul_device,
            mxfp4_execution=self.mxfp4_execution,
            hf_triton_module_cache_mb=max(0, int(self.hf_triton_module_cache_mb)),
            hf_triton_module_cache_min_requests=self.hf_triton_module_cache_min_requests,
            hf_triton_shared_module_cache=self._hf_triton_module_cache,
            hf_triton_shared_module_cache_state=self._hf_triton_module_cache_state,
            hf_triton_shared_module_request_counts=self._hf_triton_module_request_counts,
            max_vram_bytes=self.max_vram_bytes,
            sync_cuda_timing=self.sync_cuda_timing,
        )
        original_forward = module.forward

        def selective_forward(module_self, hidden_states, router_indices=None, routing_weights=None):
            if router_indices is None or routing_weights is None:
                raise ValueError("GPT-OSS selective fused expert adapter requires router_indices and routing_weights.")
            return module_self._airllm_gpt_oss_adapter.forward(hidden_states, router_indices, routing_weights)

        module._airllm_original_forward = original_forward
        module._airllm_gpt_oss_adapter = adapter
        module.forward = types.MethodType(selective_forward, module)
        self._expert_modules[shard_prefix] = module
        return True

    def _maybe_patch_qwen35_fused_experts(self, layer_name, module_name, module):
        if self._moe_manifest.get("adapter_name") != "qwen3_5_moe":
            return False
        if not self._is_qwen35_experts_module(module):
            return False

        shard_prefix = f"{layer_name}.{module_name}" if module_name else layer_name
        owner = self

        def expert_loader(expert_id):
            return owner._load_qwen35_expert_state_dict(f"{shard_prefix}.{int(expert_id)}")

        def batch_expert_loader(expert_ids):
            return owner._load_qwen35_expert_state_dicts(shard_prefix, expert_ids)

        adapter = Qwen35SelectiveFusedMoEAdapter(
            router=None,
            expert_loader=expert_loader,
            batch_expert_loader=batch_expert_loader if self.enable_qwen35_batch_direct_slice else None,
            num_experts=module.num_experts,
            top_k=self._qwen35_config_value("num_experts_per_tok", getattr(module, "top_k", 8)),
            cache_size=0,
            expert_execution_mode=self.expert_execution_mode,
            layer_name=layer_name,
        )
        original_forward = module.forward

        def selective_forward(module_self, hidden_states, top_k_index, top_k_weights):
            return module_self._airllm_qwen35_adapter.forward(hidden_states, top_k_index, top_k_weights)

        module._airllm_original_forward = original_forward
        module._airllm_qwen35_adapter = adapter
        module.forward = types.MethodType(selective_forward, module)
        self._expert_modules[shard_prefix] = module
        return True

    def _maybe_patch_timed_module(self, layer_name, module_name, module):
        if getattr(module, "_airllm_timing_patched", False):
            return
        stat_key = None
        if module_name == "self_attn" or module_name.endswith(".self_attn"):
            stat_key = "attention_seconds"
        elif module_name in {"mlp.router", "router"} or module_name.endswith(".router"):
            stat_key = "router_seconds"
        elif module_name == "mlp" or module_name.endswith(".mlp"):
            stat_key = "mlp_total_seconds"
        if stat_key is None:
            return

        original_forward = module.forward
        owner = self

        def timed_forward(module_self, *args, **kwargs):
            owner._sync_timing_device()
            started = time.perf_counter()
            result = module_self._airllm_timing_original_forward(*args, **kwargs)
            owner._sync_timing_device()
            elapsed = time.perf_counter() - started
            layer_stats = owner._timed_module_stats.setdefault(layer_name, {})
            layer_stats[stat_key] = layer_stats.get(stat_key, 0.0) + elapsed
            layer_stats[f"{stat_key}_calls"] = layer_stats.get(f"{stat_key}_calls", 0) + 1
            input_device = owner._first_tensor_device(args, kwargs)
            output_device = owner._first_tensor_device((result,), {})
            if input_device is not None:
                layer_stats[f"{stat_key}_input_device"] = input_device
            if output_device is not None:
                layer_stats[f"{stat_key}_output_device"] = output_device
            return result

        module._airllm_timing_patched = True
        module._airllm_timing_original_forward = original_forward
        module.forward = types.MethodType(timed_forward, module)

    def _sync_timing_device(self):
        if self.sync_cuda_timing and self.running_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(self.running_device))

    @staticmethod
    def _first_tensor_device(args, kwargs):
        def visit(value):
            if isinstance(value, torch.Tensor):
                return str(value.device)
            if isinstance(value, (list, tuple)):
                for item in value:
                    found = visit(item)
                    if found is not None:
                        return found
            if isinstance(value, dict):
                for item in value.values():
                    found = visit(item)
                    if found is not None:
                        return found
            return None

        found = visit(args)
        if found is not None:
            return found
        return visit(kwargs)

    @staticmethod
    def _is_gpt_oss_experts_module(module):
        return all(
            hasattr(module, attr)
            for attr in ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias", "num_experts")
        )

    @staticmethod
    def _is_qwen35_experts_module(module):
        return all(
            hasattr(module, attr)
            for attr in ("gate_up_proj", "down_proj", "num_experts")
        ) and not hasattr(module, "gate_up_proj_bias")

    def _qwen35_config_value(self, name, default=None):
        if hasattr(self.config, name):
            return getattr(self.config, name)
        text_config = getattr(self.config, "text_config", None)
        if text_config is not None and hasattr(text_config, name):
            return getattr(text_config, name)
        return default

    @staticmethod
    def _iter_experts(module_name, module):
        if "expert" not in module_name.lower():
            return []
        if isinstance(module, nn.ModuleList):
            return [(str(idx), expert) for idx, expert in enumerate(module) if isinstance(expert, nn.Module)]
        if isinstance(module, nn.ModuleDict):
            return [(str(key), expert) for key, expert in module.items() if isinstance(expert, nn.Module)]
        return []

    def _patch_expert_forward(self, expert, shard_name):
        if getattr(expert, "_airllm_lazy_expert_patched", False):
            expert._airllm_expert_shard_name = shard_name
            expert._airllm_owner = self
            return

        original_forward = expert.forward
        owner = self

        def lazy_forward(expert_self, *args, **kwargs):
            current_owner = expert_self._airllm_owner
            if current_owner._expert_needs_loading(expert_self):
                current_owner._expert_cache_misses += 1
                state_dict = current_owner._load_expert_state_dict(expert_self._airllm_expert_shard_name)
                current_owner.move_layer_to_device(state_dict)
                current_owner._remember_resident_expert(expert_self._airllm_expert_shard_name, expert_self)
            else:
                current_owner._expert_cache_hits += 1
                current_owner._touch_resident_expert(expert_self._airllm_expert_shard_name)
            return expert_self._airllm_original_forward(*args, **kwargs)

        expert._airllm_lazy_expert_patched = True
        expert._airllm_owner = owner
        expert._airllm_expert_shard_name = shard_name
        expert._airllm_original_forward = original_forward
        expert.forward = types.MethodType(lazy_forward, expert)

    @staticmethod
    def _expert_needs_loading(expert):
        for parameter in expert.parameters(recurse=True):
            return parameter.device.type == "meta"
        for buffer in expert.buffers(recurse=True):
            return buffer.device.type == "meta"
        return False

    def offload_layer_from_device(self, layer, moved_layers):
        for param_name in moved_layers:
            set_module_tensor_to_device(self.model, param_name, 'meta')
        self._enforce_resident_expert_budget()

    def _remember_resident_expert(self, shard_name, expert):
        expert_size = self._module_nbytes(expert)
        previous_size = self._resident_expert_cache.pop(shard_name, None)
        if previous_size is not None:
            self._resident_expert_bytes -= previous_size

        self._resident_expert_cache[shard_name] = expert_size
        self._resident_expert_bytes += expert_size
        self._enforce_resident_expert_budget()

    def _touch_resident_expert(self, shard_name):
        if shard_name in self._resident_expert_cache:
            self._resident_expert_cache.move_to_end(shard_name)

    def _enforce_resident_expert_budget(self):
        if self._resident_expert_budget_bytes <= 0:
            while self._resident_expert_cache:
                self._evict_oldest_expert()
            return

        while self._resident_expert_bytes > self._resident_expert_budget_bytes and self._resident_expert_cache:
            self._evict_oldest_expert()

    def _evict_oldest_expert(self):
        shard_name, expert_size = self._resident_expert_cache.popitem(last=False)
        expert = self._expert_modules.get(shard_name)
        if expert is not None:
            self._offload_expert(expert)
        self._resident_expert_bytes -= expert_size
        clean_memory()

    def _offload_expert(self, expert):
        for param_name, _ in expert.named_parameters(recurse=True):
            set_module_tensor_to_device(expert, param_name, 'meta')
        for buffer_name, _ in expert.named_buffers(recurse=True):
            set_module_tensor_to_device(expert, buffer_name, 'meta')

    @staticmethod
    def _module_nbytes(module):
        total = 0
        for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
            if tensor.device.type == "meta":
                continue
            total += tensor.numel() * tensor.element_size()
        return total

    def _resolve_expert_cache_budget(self, explicit_mb):
        if explicit_mb is not None:
            return max(0, int(explicit_mb)) * 1024 * 1024
        if not self.running_device.startswith("cuda") or not torch.cuda.is_available():
            return 0

        device = torch.device(self.running_device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserve = 2 * 1024 * 1024 * 1024
        budget = max(0, total_memory - reserve)
        budget = min(budget // 4, 4 * 1024 * 1024 * 1024)
        return max(0, budget)

    def _resolve_gpu_packed_expert_cache_budget(self):
        mode = str(self.vram_cache_mode or "off").lower()
        if mode not in {"packed_experts", "hybrid"}:
            return 0
        if not self.running_device.startswith("cuda") or not torch.cuda.is_available():
            return 0
        requested = max(0, int(self.vram_cache_mb or 0)) * 1024 * 1024
        if requested <= 0:
            return 0
        if self.max_vram_bytes is not None:
            reserve = 512 * 1024 * 1024
            budget = max(0, int(self.max_vram_bytes) - reserve)
            return min(requested, budget)
        return requested

    def _resolve_gpu_dense_expert_cache_budget(self):
        mode = str(self.vram_cache_mode or "off").lower()
        allowed_modes = {"dense_layers", "hybrid"}
        if getattr(self, "_qwen35_moe_layout", False):
            allowed_modes.add("packed_experts")
        if mode not in allowed_modes:
            return 0
        if not self.running_device.startswith("cuda") or not torch.cuda.is_available():
            return 0

        requested_mb = self.dense_expert_cache_mb
        if requested_mb is None:
            if mode == "dense_layers" or getattr(self, "_qwen35_moe_layout", False):
                requested_mb = self.vram_cache_mb
            else:
                requested_mb = 0
        requested = max(0, int(requested_mb or 0)) * 1024 * 1024
        if requested <= 0:
            return 0
        if self.max_vram_bytes is not None:
            reserve = 512 * 1024 * 1024
            packed_budget = int(getattr(self, "_gpu_packed_expert_budget_bytes", 0) or 0)
            budget = max(0, int(self.max_vram_bytes) - reserve - packed_budget)
            return min(requested, budget)
        return requested

    @staticmethod
    def _is_packed_gpt_oss_state_dict(state_dict):
        return {
            "gate_up_proj_blocks",
            "gate_up_proj_scales",
            "gate_up_proj_bias",
            "down_proj_blocks",
            "down_proj_scales",
            "down_proj_bias",
        }.issubset(state_dict)

    def _gpu_packed_cache_device(self):
        if not torch.cuda.is_available():
            return None
        for candidate in (self.mxfp4_device, self.expert_matmul_device, self.dense_device, self.running_device):
            if candidate and str(candidate).startswith("cuda"):
                return torch.device(candidate)
        return None

    def _maybe_cache_gpu_packed_expert(self, shard_name, state_dict):
        if self._gpu_packed_expert_budget_bytes <= 0:
            return
        if not self._is_packed_gpt_oss_state_dict(state_dict):
            return
        device = self._gpu_packed_cache_device()
        if device is None:
            return
        state_bytes = self._state_dict_nbytes(state_dict)
        if state_bytes > self._gpu_packed_expert_budget_bytes:
            return
        if shard_name in self._gpu_packed_expert_cache:
            _, previous_bytes = self._gpu_packed_expert_cache.pop(shard_name)
            self._gpu_packed_expert_bytes -= previous_bytes
        cached = {
            key: tensor.to(device=device, non_blocking=True).contiguous()
            for key, tensor in state_dict.items()
        }
        cached_bytes = self._state_dict_nbytes(cached)
        self._gpu_packed_expert_cache[shard_name] = (cached, cached_bytes)
        self._gpu_packed_expert_bytes += cached_bytes
        self._enforce_gpu_packed_expert_budget()

    def _get_gpu_packed_expert(self, shard_name):
        if self._gpu_packed_expert_budget_bytes <= 0:
            return None
        if shard_name not in self._gpu_packed_expert_cache:
            self._gpu_packed_expert_cache_misses += 1
            return None
        state_dict, state_dict_bytes = self._gpu_packed_expert_cache.pop(shard_name)
        self._gpu_packed_expert_cache[shard_name] = (state_dict, state_dict_bytes)
        self._gpu_packed_expert_cache_hits += 1
        self._gpu_packed_expert_bytes_avoided += state_dict_bytes
        return state_dict

    def _enforce_gpu_packed_expert_budget(self):
        while self._gpu_packed_expert_bytes > self._gpu_packed_expert_budget_bytes and self._gpu_packed_expert_cache:
            _, (_, state_dict_bytes) = self._gpu_packed_expert_cache.popitem(last=False)
            self._gpu_packed_expert_bytes -= state_dict_bytes

    @staticmethod
    def _is_dense_gpt_oss_state_dict(state_dict):
        return {
            "gate_up_proj",
            "gate_up_proj_bias",
            "down_proj",
            "down_proj_bias",
        }.issubset(state_dict)

    @staticmethod
    def _is_qwen35_state_dict(state_dict):
        return {
            "gate_up_proj.weight",
            "down_proj.weight",
        }.issubset(state_dict)

    def _gpu_dense_cache_device(self):
        if not torch.cuda.is_available():
            return None
        for candidate in (self.expert_matmul_device, self.mxfp4_device, self.dense_device, self.running_device):
            if candidate and str(candidate).startswith("cuda"):
                return torch.device(candidate)
        return None

    def _dense_cache_compute_dtype(self):
        dtype = getattr(self, "dtype", None)
        if dtype is torch.float16:
            return torch.float16
        if dtype is torch.float32:
            return torch.float32
        return torch.float16

    def _get_gpu_dense_expert(self, shard_name):
        if self._gpu_dense_expert_budget_bytes <= 0:
            return None
        if shard_name not in self._gpu_dense_expert_cache:
            self._gpu_dense_expert_cache_misses += 1
            return None
        state_dict, state_dict_bytes = self._gpu_dense_expert_cache.pop(shard_name)
        self._gpu_dense_expert_cache[shard_name] = (state_dict, state_dict_bytes)
        self._gpu_dense_expert_cache_hits += 1
        self._gpu_dense_expert_bytes_avoided += state_dict_bytes
        return state_dict

    def _maybe_cache_gpu_dense_expert(self, shard_name, state_dict):
        if self._gpu_dense_expert_budget_bytes <= 0:
            return None
        if self._gpu_dense_expert_disabled_no_hits:
            self._gpu_dense_expert_admission_skips += 1
            return None
        if (
            self.dense_expert_cache_max_builds_without_hit > 0
            and self._gpu_dense_expert_cache_hits == 0
            and self._gpu_dense_expert_builds >= self.dense_expert_cache_max_builds_without_hit
        ):
            self._gpu_dense_expert_disabled_no_hits = True
            self._gpu_dense_expert_admission_skips += 1
            return None
        request_count = self._gpu_dense_expert_request_counts.get(shard_name, 0) + 1
        self._gpu_dense_expert_request_counts[shard_name] = request_count
        if request_count < max(1, int(self.dense_expert_cache_min_requests)):
            self._gpu_dense_expert_admission_skips += 1
            return None
        if self._is_dense_gpt_oss_state_dict(state_dict) or self._is_qwen35_state_dict(state_dict):
            dense_state = state_dict
        elif self._is_packed_gpt_oss_state_dict(state_dict):
            device = self._gpu_dense_cache_device()
            if device is None:
                return None
            started = time.perf_counter()
            dense_state = dequantize_mxfp4_expert(
                state_dict,
                dtype=self._dense_cache_compute_dtype(),
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self._gpu_dense_expert_build_seconds += time.perf_counter() - started
            self._gpu_dense_expert_builds += 1
        else:
            return None

        device = self._gpu_dense_cache_device()
        if device is None:
            return None
        dense_state = {
            key: tensor.to(device=device, non_blocking=True).contiguous()
            for key, tensor in dense_state.items()
        }
        state_bytes = self._state_dict_nbytes(dense_state)
        if state_bytes > self._gpu_dense_expert_budget_bytes:
            return None
        if shard_name in self._gpu_dense_expert_cache:
            _, previous_bytes = self._gpu_dense_expert_cache.pop(shard_name)
            self._gpu_dense_expert_bytes -= previous_bytes
        self._gpu_dense_expert_cache[shard_name] = (dense_state, state_bytes)
        self._gpu_dense_expert_bytes += state_bytes
        self._enforce_gpu_dense_expert_budget()
        return dense_state

    def _enforce_gpu_dense_expert_budget(self):
        while self._gpu_dense_expert_bytes > self._gpu_dense_expert_budget_bytes and self._gpu_dense_expert_cache:
            _, (_, state_dict_bytes) = self._gpu_dense_expert_cache.popitem(last=False)
            self._gpu_dense_expert_bytes -= state_dict_bytes

    def _load_expert_state_dict(self, shard_name):
        if shard_name in self._cpu_expert_cache:
            state_dict, state_dict_bytes = self._cpu_expert_cache.pop(shard_name)
            self._cpu_expert_cache[shard_name] = (state_dict, state_dict_bytes)
            self._cpu_expert_cache_hits += 1
            return state_dict

        self._cpu_expert_cache_misses += 1
        state_dict = self.load_layer_to_cpu(shard_name)
        if self._cpu_expert_budget_bytes > 0:
            state_dict_bytes = self._state_dict_nbytes(state_dict)
            self._cpu_expert_cache[shard_name] = (state_dict, state_dict_bytes)
            self._cpu_expert_bytes += state_dict_bytes
            self._enforce_cpu_expert_budget()
        return state_dict

    def _load_gpt_oss_expert_state_dict(self, shard_name):
        dense_cached = self._get_gpu_dense_expert(shard_name)
        if dense_cached is not None:
            return dense_cached

        gpu_cached = self._get_gpu_packed_expert(shard_name)
        if gpu_cached is not None:
            dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, gpu_cached)
            if dense_cached is not None:
                return dense_cached
            return gpu_cached

        if self.expert_materialization_mode in {"direct_slice", "hybrid"} and self._lazy_split_builder is not None:
            if shard_name in self._cpu_expert_cache:
                state_dict, _ = self._cpu_expert_cache.pop(shard_name)
                self._cpu_expert_cache[shard_name] = (state_dict, self._state_dict_nbytes(state_dict))
                self._cpu_expert_cache_hits += 1
                self._maybe_cache_gpu_packed_expert(shard_name, state_dict)
                dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, state_dict)
                return dense_cached if dense_cached is not None else state_dict
            self._cpu_expert_cache_misses += 1
            self._expert_cache_misses += 1
            state_dict = self._lazy_split_builder.load_gpt_oss_expert_direct(shard_name)
            self._maybe_cache_gpu_packed_expert(shard_name, state_dict)
            dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, state_dict)
            if self._cpu_expert_budget_bytes > 0:
                state_dict_bytes = self._state_dict_nbytes(state_dict)
                self._cpu_expert_cache[shard_name] = (state_dict, state_dict_bytes)
                self._cpu_expert_bytes += state_dict_bytes
                self._enforce_cpu_expert_budget()
            return dense_cached if dense_cached is not None else state_dict

        state_dict = self._load_expert_state_dict(shard_name)
        mapped = self._map_gpt_oss_expert_state_dict(shard_name, state_dict)
        self._maybe_cache_gpu_packed_expert(shard_name, mapped)
        dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, mapped)
        return dense_cached if dense_cached is not None else mapped

    def _load_gpt_oss_expert_state_dicts(self, shard_prefix, expert_ids):
        result = {}
        missing_shard_names = []
        for expert_id in sorted({int(expert_id) for expert_id in expert_ids}):
            shard_name = f"{shard_prefix}.{expert_id}"
            dense_cached = self._get_gpu_dense_expert(shard_name)
            if dense_cached is not None:
                result[expert_id] = dense_cached
                continue
            gpu_cached = self._get_gpu_packed_expert(shard_name)
            if gpu_cached is not None:
                dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, gpu_cached)
                result[expert_id] = dense_cached if dense_cached is not None else gpu_cached
                continue
            if shard_name in self._cpu_expert_cache:
                state_dict, _ = self._cpu_expert_cache.pop(shard_name)
                self._cpu_expert_cache[shard_name] = (state_dict, self._state_dict_nbytes(state_dict))
                self._cpu_expert_cache_hits += 1
                self._maybe_cache_gpu_packed_expert(shard_name, state_dict)
                dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, state_dict)
                result[expert_id] = dense_cached if dense_cached is not None else state_dict
            else:
                missing_shard_names.append(shard_name)

        if missing_shard_names:
            self._cpu_expert_cache_misses += len(missing_shard_names)
            self._expert_cache_misses += len(missing_shard_names)
            if self.expert_materialization_mode in {"direct_slice", "hybrid"} and self._lazy_split_builder is not None:
                loaded = self._lazy_split_builder.load_gpt_oss_experts_direct(missing_shard_names)
            else:
                loaded = {
                    shard_name: self._map_gpt_oss_expert_state_dict(shard_name, self._load_expert_state_dict(shard_name))
                    for shard_name in missing_shard_names
                }
            for shard_name in missing_shard_names:
                expert_id = int(shard_name.rsplit(".", 1)[1])
                state_dict = loaded[shard_name]
                self._maybe_cache_gpu_packed_expert(shard_name, state_dict)
                dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, state_dict)
                result[expert_id] = dense_cached if dense_cached is not None else state_dict
                if self._cpu_expert_budget_bytes > 0:
                    state_dict_bytes = self._state_dict_nbytes(state_dict)
                    self._cpu_expert_cache[shard_name] = (state_dict, state_dict_bytes)
                    self._cpu_expert_bytes += state_dict_bytes
                    self._enforce_cpu_expert_budget()
        return result

    def _load_qwen35_expert_state_dict(self, shard_name):
        dense_cached = self._get_gpu_dense_expert(shard_name)
        if dense_cached is not None:
            return dense_cached

        if self.expert_materialization_mode in {"direct_slice", "hybrid"} and self._lazy_split_builder is not None:
            if shard_name in self._cpu_expert_cache:
                state_dict, _ = self._cpu_expert_cache.pop(shard_name)
                self._cpu_expert_cache[shard_name] = (state_dict, self._state_dict_nbytes(state_dict))
                self._cpu_expert_cache_hits += 1
                dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, state_dict)
                return dense_cached if dense_cached is not None else state_dict
            self._cpu_expert_cache_misses += 1
            self._expert_cache_misses += 1
            state_dict = self._lazy_split_builder.load_qwen35_expert_direct(shard_name)
            dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, state_dict)
            if self._cpu_expert_budget_bytes > 0:
                state_dict_bytes = self._state_dict_nbytes(state_dict)
                self._cpu_expert_cache[shard_name] = (state_dict, state_dict_bytes)
                self._cpu_expert_bytes += state_dict_bytes
                self._enforce_cpu_expert_budget()
            return dense_cached if dense_cached is not None else state_dict

        state_dict = self._load_expert_state_dict(shard_name)
        mapped = self._map_qwen35_expert_state_dict(shard_name, state_dict)
        dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, mapped)
        return dense_cached if dense_cached is not None else mapped

    def _load_qwen35_expert_state_dicts(self, shard_prefix, expert_ids):
        result = {}
        missing_shard_names = []
        for expert_id in sorted({int(expert_id) for expert_id in expert_ids}):
            shard_name = f"{shard_prefix}.{expert_id}"
            dense_cached = self._get_gpu_dense_expert(shard_name)
            if dense_cached is not None:
                result[expert_id] = dense_cached
                continue
            if shard_name in self._cpu_expert_cache:
                state_dict, _ = self._cpu_expert_cache.pop(shard_name)
                self._cpu_expert_cache[shard_name] = (state_dict, self._state_dict_nbytes(state_dict))
                self._cpu_expert_cache_hits += 1
                dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, state_dict)
                result[expert_id] = dense_cached if dense_cached is not None else state_dict
            else:
                missing_shard_names.append(shard_name)

        if missing_shard_names:
            self._cpu_expert_cache_misses += len(missing_shard_names)
            self._expert_cache_misses += len(missing_shard_names)
            if self.expert_materialization_mode in {"direct_slice", "hybrid"} and self._lazy_split_builder is not None:
                loaded = self._lazy_split_builder.load_qwen35_experts_direct(missing_shard_names)
            else:
                loaded = {
                    shard_name: self._map_qwen35_expert_state_dict(shard_name, self._load_expert_state_dict(shard_name))
                    for shard_name in missing_shard_names
                }
            for shard_name in missing_shard_names:
                expert_id = int(shard_name.rsplit(".", 1)[1])
                state_dict = loaded[shard_name]
                dense_cached = self._maybe_cache_gpu_dense_expert(shard_name, state_dict)
                result[expert_id] = dense_cached if dense_cached is not None else state_dict
                if self._cpu_expert_budget_bytes > 0:
                    state_dict_bytes = self._state_dict_nbytes(state_dict)
                    self._cpu_expert_cache[shard_name] = (state_dict, state_dict_bytes)
                    self._cpu_expert_bytes += state_dict_bytes
                    self._enforce_cpu_expert_budget()
        return result

    @staticmethod
    def _map_gpt_oss_expert_state_dict(shard_name, state_dict):
        mapped = {}
        for key, tensor in state_dict.items():
            if key.endswith(".gate_up_proj.weight"):
                mapped["gate_up_proj"] = tensor
            elif key.endswith(".gate_up_proj_bias.weight"):
                mapped["gate_up_proj_bias"] = tensor
            elif key.endswith(".down_proj.weight"):
                mapped["down_proj"] = tensor
            elif key.endswith(".down_proj_bias.weight"):
                mapped["down_proj_bias"] = tensor
            elif key.endswith(".gate_up_proj"):
                mapped["gate_up_proj"] = tensor
            elif key.endswith(".gate_up_proj_bias"):
                mapped["gate_up_proj_bias"] = tensor
            elif key.endswith(".down_proj"):
                mapped["down_proj"] = tensor
            elif key.endswith(".down_proj_bias"):
                mapped["down_proj_bias"] = tensor
            elif key.endswith(".gate_up_proj_blocks.weight") or key.endswith(".gate_up_proj_blocks"):
                mapped["gate_up_proj_blocks"] = tensor
            elif key.endswith(".gate_up_proj_scales.weight") or key.endswith(".gate_up_proj_scales"):
                mapped["gate_up_proj_scales"] = tensor
            elif key.endswith(".down_proj_blocks.weight") or key.endswith(".down_proj_blocks"):
                mapped["down_proj_blocks"] = tensor
            elif key.endswith(".down_proj_scales.weight") or key.endswith(".down_proj_scales"):
                mapped["down_proj_scales"] = tensor

        if {"gate_up_proj_blocks", "gate_up_proj_scales", "down_proj_blocks", "down_proj_scales"}.issubset(mapped):
            missing = {"gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias",
                       "down_proj_blocks", "down_proj_scales", "down_proj_bias"} - set(mapped)
            if missing:
                raise KeyError(f"GPT-OSS MXFP4 expert shard {shard_name} missing tensors: {sorted(missing)}")
            return mapped

        missing = {"gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"} - set(mapped)
        if missing:
            raise KeyError(f"GPT-OSS expert shard {shard_name} missing tensors: {sorted(missing)}")
        return mapped

    @staticmethod
    def _map_qwen35_expert_state_dict(shard_name, state_dict):
        mapped = {}
        for key, tensor in state_dict.items():
            if key.endswith(".gate_up_proj.weight") or key.endswith(".gate_up_proj"):
                mapped["gate_up_proj.weight"] = tensor
            elif key.endswith(".down_proj.weight") or key.endswith(".down_proj"):
                mapped["down_proj.weight"] = tensor

        missing = {"gate_up_proj.weight", "down_proj.weight"} - set(mapped)
        if missing:
            raise KeyError(f"Qwen3.6 expert shard {shard_name} missing tensors: {sorted(missing)}")
        return mapped

    def _enforce_cpu_expert_budget(self):
        while self._cpu_expert_bytes > self._cpu_expert_budget_bytes and self._cpu_expert_cache:
            _, (_, state_dict_bytes) = self._cpu_expert_cache.popitem(last=False)
            self._cpu_expert_bytes -= state_dict_bytes

    @staticmethod
    def _state_dict_nbytes(state_dict):
        total = 0
        for tensor in state_dict.values():
            if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                total += tensor.numel() * tensor.element_size()
        return total

    def moe_cache_stats(self):
        total = self._expert_cache_hits + self._expert_cache_misses
        hit_rate = 0.0 if total == 0 else self._expert_cache_hits / total
        cpu_total = self._cpu_expert_cache_hits + self._cpu_expert_cache_misses
        cpu_hit_rate = 0.0 if cpu_total == 0 else self._cpu_expert_cache_hits / cpu_total
        fused_adapter_stats = self._fused_adapter_stats()
        split_builder_status = self._lazy_split_builder.status() if self._lazy_split_builder is not None else {}
        return {
            "resident_experts": len(self._resident_expert_cache),
            "resident_expert_bytes": self._resident_expert_bytes,
            "resident_expert_budget_bytes": self._resident_expert_budget_bytes,
            "expert_cache_hits": self._expert_cache_hits,
            "expert_cache_misses": self._expert_cache_misses,
            "expert_cache_hit_rate": hit_rate,
            "cpu_cached_experts": len(self._cpu_expert_cache),
            "cpu_expert_bytes": self._cpu_expert_bytes,
            "cpu_expert_budget_bytes": self._cpu_expert_budget_bytes,
            "cpu_expert_cache_hits": self._cpu_expert_cache_hits,
            "cpu_expert_cache_misses": self._cpu_expert_cache_misses,
            "cpu_expert_cache_hit_rate": cpu_hit_rate,
            "vram_cache_mode": self.vram_cache_mode,
            "expert_execution_mode": self.expert_execution_mode,
            "runtime_mxfp4_execution": self.mxfp4_execution,
            "hf_triton_module_cache_mb": self.hf_triton_module_cache_mb,
            "hf_triton_module_cache_min_requests": self.hf_triton_module_cache_min_requests,
            "hf_triton_cached_modules": len(self._hf_triton_module_cache),
            "hf_triton_module_unique_requests_global": len(self._hf_triton_module_request_counts),
            "hf_triton_module_cache_bytes_global": int(self._hf_triton_module_cache_state.get("bytes", 0)),
            "hf_triton_module_cache_budget_bytes_global": int(self._hf_triton_module_cache_state.get("budget_bytes", 0)),
            "gpu_packed_cached_experts": len(self._gpu_packed_expert_cache),
            "gpu_packed_expert_bytes": self._gpu_packed_expert_bytes,
            "gpu_packed_expert_budget_bytes": self._gpu_packed_expert_budget_bytes,
            "gpu_packed_expert_cache_hits": self._gpu_packed_expert_cache_hits,
            "gpu_packed_expert_cache_misses": self._gpu_packed_expert_cache_misses,
            "gpu_packed_expert_bytes_avoided": self._gpu_packed_expert_bytes_avoided,
            "gpu_dense_cached_experts": len(self._gpu_dense_expert_cache),
            "gpu_dense_expert_bytes": self._gpu_dense_expert_bytes,
            "gpu_dense_expert_budget_bytes": self._gpu_dense_expert_budget_bytes,
            "gpu_dense_expert_cache_hits": self._gpu_dense_expert_cache_hits,
            "gpu_dense_expert_cache_misses": self._gpu_dense_expert_cache_misses,
            "gpu_dense_expert_bytes_avoided": self._gpu_dense_expert_bytes_avoided,
            "gpu_dense_expert_builds": self._gpu_dense_expert_builds,
            "gpu_dense_expert_build_seconds": self._gpu_dense_expert_build_seconds,
            "gpu_dense_expert_admission_skips": self._gpu_dense_expert_admission_skips,
            "gpu_dense_expert_unique_requests": len(self._gpu_dense_expert_request_counts),
            "gpu_dense_expert_min_requests": self.dense_expert_cache_min_requests,
            "gpu_dense_expert_max_builds_without_hit": self.dense_expert_cache_max_builds_without_hit,
            "gpu_dense_expert_disabled_no_hits": self._gpu_dense_expert_disabled_no_hits,
            "dense_layer_cache_hits": self._gpu_dense_expert_cache_hits,
            "dense_layer_cache_misses": self._gpu_dense_expert_cache_misses,
            "manifest_expert_count": self._moe_manifest.get("expert_count", 0),
            "manifest_direct_expert_count": self._moe_manifest.get("direct_expert_count", 0),
            "manifest_fused_tensor_count": self._moe_manifest.get("fused_tensor_count", 0),
            "selective_fused_runtime": self._moe_manifest.get("selective_fused_runtime", False),
            "adapter_name": self._moe_manifest.get("adapter_name"),
            "mxfp4_execution": self._moe_manifest.get("mxfp4_execution"),
            "dense_contains_full_fused_experts": self._moe_manifest.get("dense_contains_full_fused_experts"),
            "requires_dense_fallback": self._moe_manifest.get("requires_dense_fallback"),
            "strict_full_fused_tensor_loaded": bool(self._moe_manifest.get("dense_contains_full_fused_experts")),
            "fused_adapter_stats": fused_adapter_stats,
            "timed_module_stats": self._timed_module_stats,
            "packed_expert_bytes_loaded": fused_adapter_stats.get("packed_expert_bytes_loaded", 0),
            "run_expert_calls": fused_adapter_stats.get("run_expert_calls", 0),
            "grouped_expert_calls": fused_adapter_stats.get("grouped_expert_calls", 0),
            "per_token_expert_calls": fused_adapter_stats.get("per_token_expert_calls", 0),
            "expert_input_rows": fused_adapter_stats.get("expert_input_rows", 0),
            "dequantized_temporary_bytes": fused_adapter_stats.get("dequantized_temporary_bytes", 0),
            "mxfp4_dequant_seconds": fused_adapter_stats.get("mxfp4_dequant_seconds", 0.0),
            "hf_triton_kernel_seconds": fused_adapter_stats.get("hf_triton_kernel_seconds", 0.0),
            "hf_triton_module_cache_hits": fused_adapter_stats.get("hf_triton_module_cache_hits", 0),
            "hf_triton_module_cache_misses": fused_adapter_stats.get("hf_triton_module_cache_misses", 0),
            "hf_triton_module_builds": fused_adapter_stats.get("hf_triton_module_builds", 0),
            "hf_triton_module_build_seconds": fused_adapter_stats.get("hf_triton_module_build_seconds", 0.0),
            "hf_triton_module_cache_bytes": int(self._hf_triton_module_cache_state.get("bytes", 0)),
            "hf_triton_module_cache_budget_bytes": int(self._hf_triton_module_cache_state.get("budget_bytes", 0)),
            "hf_triton_module_cache_evictions": fused_adapter_stats.get("hf_triton_module_cache_evictions", 0),
            "hf_triton_module_admission_skips": fused_adapter_stats.get("hf_triton_module_admission_skips", 0),
            "hf_triton_module_unique_requests": len(self._hf_triton_module_request_counts),
            "hf_triton_module_direct_loads_skipped": fused_adapter_stats.get("hf_triton_module_direct_loads_skipped", 0),
            "hf_triton_fallbacks": fused_adapter_stats.get("hf_triton_fallbacks", 0),
            "expert_matmul_seconds": fused_adapter_stats.get("expert_matmul_seconds", 0.0),
            "gate_up_matmul_seconds": fused_adapter_stats.get("gate_up_matmul_seconds", 0.0),
            "down_matmul_seconds": fused_adapter_stats.get("down_matmul_seconds", 0.0),
            "activation_seconds": fused_adapter_stats.get("activation_seconds", 0.0),
            "expert_materialization_mode": self.expert_materialization_mode,
            "split_builder_status": split_builder_status,
            "selected_expert_builds_this_run": split_builder_status.get("selected_expert_builds_this_run", 0),
            "unused_expert_builds_this_run": split_builder_status.get("unused_expert_builds_this_run", 0),
            "direct_slice_expert_loads_this_run": split_builder_status.get("direct_slice_expert_loads_this_run", 0),
            "direct_slice_packed_bytes_read_this_run": split_builder_status.get("direct_slice_packed_bytes_read_this_run", 0),
            "selected_expert_materialization_time_this_run": split_builder_status.get("selected_expert_materialization_time_this_run", 0.0),
        }

    def _fused_adapter_stats(self):
        totals = {}
        for module in self._expert_modules.values():
            adapter = getattr(module, "_airllm_gpt_oss_adapter", None) or getattr(module, "_airllm_qwen35_adapter", None)
            if adapter is None:
                continue
            for key, value in adapter.stats.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
                elif key == "layer_timing_breakdown" and isinstance(value, dict):
                    layer_totals = totals.setdefault(key, {})
                    for layer_name, layer_stats in value.items():
                        merged = layer_totals.setdefault(layer_name, {})
                        for stat_key, stat_value in layer_stats.items():
                            if isinstance(stat_value, (int, float)):
                                merged[stat_key] = merged.get(stat_key, 0) + stat_value
                            elif stat_key == "selected_experts":
                                existing = set(merged.get(stat_key, []))
                                existing.update(int(item) for item in stat_value)
                                merged[stat_key] = sorted(existing)
                                merged["selected_expert_count"] = len(existing)
                elif key == "device_report" and isinstance(value, dict):
                    merged_report = totals.setdefault(key, {})
                    merged_report.update(value)
        return totals

    def reset_runtime_stats(self):
        super().reset_runtime_stats()
        self._expert_cache_hits = 0
        self._expert_cache_misses = 0
        self._cpu_expert_cache_hits = 0
        self._cpu_expert_cache_misses = 0
        self._gpu_packed_expert_cache_hits = 0
        self._gpu_packed_expert_cache_misses = 0
        self._gpu_packed_expert_bytes_avoided = 0
        self._gpu_dense_expert_cache_hits = 0
        self._gpu_dense_expert_cache_misses = 0
        self._gpu_dense_expert_bytes_avoided = 0
        self._gpu_dense_expert_builds = 0
        self._gpu_dense_expert_build_seconds = 0.0
        self._gpu_dense_expert_request_counts = {}
        self._gpu_dense_expert_admission_skips = 0
        self._gpu_dense_expert_disabled_no_hits = False
        self._timed_module_stats = {}
        if self._lazy_split_builder is not None and hasattr(self._lazy_split_builder, "reset_run_metrics"):
            self._lazy_split_builder.reset_run_metrics()
        for module in self._expert_modules.values():
            adapter = getattr(module, "_airllm_gpt_oss_adapter", None) or getattr(module, "_airllm_qwen35_adapter", None)
            if adapter is not None:
                adapter.reset_stats()

    def runtime_stats(self):
        stats = super().runtime_stats()
        stats.update(self.moe_cache_stats())
        return stats

    def flush_runtime_progress(self):
        if self._lazy_split_builder is not None and hasattr(self._lazy_split_builder, "flush_progress"):
            return self._lazy_split_builder.flush_progress()
        return {}

    def clear_runtime_caches(self):
        while self._resident_expert_cache:
            self._evict_oldest_expert()
        self._cpu_expert_cache.clear()
        self._cpu_expert_bytes = 0
        self._gpu_packed_expert_cache.clear()
        self._gpu_packed_expert_bytes = 0
        self._gpu_dense_expert_cache.clear()
        self._gpu_dense_expert_bytes = 0
        self._gpu_dense_expert_request_counts.clear()
        self.reset_runtime_stats()

    def warm_selected_expert_cache(self):
        if self._lazy_split_builder is None:
            return {"warmed_experts": 0, "reason": "no lazy split builder"}
        warmed = 0
        layer_timings = self._lazy_split_builder.progress.get("layer_timings", {})
        for layer_name, timing in layer_timings.items():
            selected = timing.get("selected_experts", {}) if isinstance(timing, dict) else {}
            for expert_id in selected:
                shard_name = f"{layer_name.rstrip('.')}.mlp.experts.{int(expert_id)}"
                if self._cpu_expert_budget_bytes > 0 and self._cpu_expert_bytes >= self._cpu_expert_budget_bytes:
                    return {"warmed_experts": warmed, "stopped": "cpu_expert_cache_budget_reached"}
                if self._moe_manifest.get("adapter_name") == "qwen3_5_moe":
                    self._load_qwen35_expert_state_dict(shard_name)
                else:
                    self._load_gpt_oss_expert_state_dict(shard_name)
                warmed += 1
        return {"warmed_experts": warmed}

    def _load_moe_manifest(self):
        manifest_path = Path(self.checkpoint_path) / "moe_expert_index.json"
        if not manifest_path.exists():
            return
        with open(manifest_path, "r", encoding="utf-8") as f:
            self._moe_manifest = json.load(f)
