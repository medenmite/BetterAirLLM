from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
import os
from concurrent.futures import ThreadPoolExecutor
from sys import platform, stderr

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, LlamaForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights

from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer, HfQuantizer

from .profiler import LayeredProfiler

try:
    from optimum.bettertransformer import BetterTransformer  # type: ignore
    _has_better_transformer = True
except ImportError:
    _has_better_transformer = False

from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path

try:
    import bitsandbytes as bnb  # type: ignore

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed', file=stderr)
except ImportError:
    bitsandbytes_installed = False


try:
    from transformers.cache_utils import Cache, DynamicCache

    cache_utils_installed = True
    print('>>>> cache_utils installed', file=stderr)
except ImportError:
    cache_utils_installed = False


class BetterAirLLMBaseModel(GenerationMixin):

    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                       'layer_prefix': 'model.layers',
                       'norm': 'model.norm',
                       'lm_head': 'lm_head',}


    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 reinitialize_each_forward=False, moe_expert_sharding=False,
                 moe_strict_streaming=False, moe_allow_dense_fallback=False,
                 moe_selective_fused_adapter_name=None, os_reserved_ram_mb=None,
                 abort_unsafe_context=False, resume_split_build=False, lazy_build=False,
                 expert_materialization_mode="persisted_split", layer_cleanup_interval=1,
                 mxfp4_device=None, expert_matmul_device=None, dense_device=None,
                 max_vram_mb=None, sync_cuda_timing=False, keep_nontransformer_resident=False,
                 quiet_progress=False):


        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self._supports_cache_class = False
        self.hf_quantizer = None
        self.reinitialize_each_forward = reinitialize_each_forward
        self.moe_expert_sharding = moe_expert_sharding
        self.moe_strict_streaming = moe_strict_streaming
        self.moe_allow_dense_fallback = moe_allow_dense_fallback
        self.moe_selective_fused_adapter_name = moe_selective_fused_adapter_name
        self.os_reserved_ram_mb = 8192 if os_reserved_ram_mb is None and platform == "win32" else (os_reserved_ram_mb or 4096)
        self.abort_unsafe_context = abort_unsafe_context
        self.resume_split_build = resume_split_build
        self.lazy_build = lazy_build
        self.expert_materialization_mode = expert_materialization_mode
        self.layer_cleanup_interval = max(0, int(layer_cleanup_interval))
        self.mxfp4_device = mxfp4_device
        self.expert_matmul_device = expert_matmul_device
        self.dense_device = dense_device
        self.max_vram_mb = max_vram_mb
        self.max_vram_bytes = None if max_vram_mb is None else int(max_vram_mb) * 1024 * 1024
        self.sync_cuda_timing = bool(sync_cuda_timing)
        self.keep_nontransformer_resident = bool(keep_nontransformer_resident)
        self.quiet_progress = bool(quiet_progress)
        self._resident_layer_names = set()
        self._lazy_split_builder = None
        self.reset_runtime_stats()

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. To use compression, please install bitsandbytes: `pip install bitsandbytes`')


        self.compression = compression
        self.hf_token = hf_token

        self.set_layer_names_dict()


        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(model_local_path_or_repo_id,
                                                                                         layer_shards_saving_path,
                                                                                         compression=compression,
                                                                                         layer_names=self.layer_names_dict,
                                                                                         hf_token=hf_token,
                                                                                         delete_original=delete_original,
                                                                                         moe_expert_sharding=moe_expert_sharding,
                                                                                         moe_strict_streaming=moe_strict_streaming,
                                                                                         moe_allow_dense_fallback=moe_allow_dense_fallback,
                                                                                         moe_selective_fused_adapter_name=moe_selective_fused_adapter_name,
                                                                                         resume_split_build=resume_split_build,
                                                                                         lazy_build=lazy_build)
        self.running_device = dense_device or device
        if str(self.running_device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({self.running_device}), but torch.cuda.is_available() is false.")
        self.device = torch.device(self.running_device)
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        if hf_token is not None:
            self.config = AutoConfig.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)

        self._supports_cache_class = self._supports_transformers_cache()
        self.max_seq_len = max_seq_len
        self.check_kv_cache_safety()

        self.generation_config = self.get_generation_config()


        self.tokenizer = self.get_tokenizer(hf_token=hf_token)


        self.init_model()
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        layers_count = len(model_attr)


        self.layer_names = [self.layer_names_dict['embed']] + [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in
                                                               range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]
        self._nontransformer_resident_layer_names = {
            self.layer_names_dict['embed'],
            self.layer_names_dict['norm'],
            self.layer_names_dict['lm_head'],
        }

        self.main_input_name = "input_ids"

        if self.moe_expert_sharding and self.lazy_build:
            from .moe_split_builder import IncrementalMoESplitBuilder

            self._lazy_split_builder = IncrementalMoESplitBuilder(
                self.model_local_path,
                self.checkpoint_path,
                layer_names=self.layer_names_dict,
                repo_id=model_local_path_or_repo_id if not Path(str(model_local_path_or_repo_id)).exists() else None,
                hf_token=hf_token,
                strict_mode=self.moe_strict_streaming,
                adapter_name=self.moe_selective_fused_adapter_name,
                allow_dense_fallback=self.moe_allow_dense_fallback,
            )

        self.prefetching = prefetching

        if self.compression is not None:
            self.prefetching = False
            print(f"not support prefetching for compression for now. loading with no prepetching mode.", file=stderr)

        if prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        if self.keep_nontransformer_resident:
            self._prepare_resident_nontransformer_layers()

    def set_experts_implementation(self, experts_implementation):
        requested = (
            experts_implementation
            if not isinstance(experts_implementation, dict)
            else experts_implementation.get("", getattr(self.config, "_experts_implementation", None))
        )
        if requested is not None and hasattr(self.config, "_experts_implementation_internal"):
            self.config._experts_implementation_internal = requested
        if hasattr(self.model, "set_experts_implementation"):
            return self.model.set_experts_implementation(experts_implementation)
        return None

    def get_generation_config(self):

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception as e:
            return GenerationConfig()

    def get_tokenizer(self, hf_token=None):
        kwargs = {"trust_remote_code": True}
        if hf_token is not None:
            kwargs["token"] = hf_token

        try:
            return AutoTokenizer.from_pretrained(self.model_local_path, **kwargs)
        except Exception:
            if getattr(self.config, "model_type", None) != "gpt_oss":
                raise
            fallback_path = self._gpt_oss_tokenizer_fallback_path()
            if fallback_path is None:
                raise
            print(f"GPT-OSS tokenizer fallback: {fallback_path}", file=stderr)
            return AutoTokenizer.from_pretrained(fallback_path, **kwargs)

    def _gpt_oss_tokenizer_fallback_path(self):
        model_path = Path(self.model_local_path)
        candidates = []
        env_path = os.environ.get("BETTERAIRLLM_GPT_OSS_TOKENIZER_PATH")
        if env_path:
            candidates.append(Path(env_path))
        if model_path.exists():
            candidates.extend([
                model_path.parent / "gpt-oss-20b-hf",
                model_path.parent / "gpt-oss-120b-hf",
            ])
        for candidate in candidates:
            if candidate == model_path:
                continue
            if (candidate / "tokenizer.json").exists():
                return str(candidate)
        return None

    def get_use_better_transformer(self):
        return True

    def init_model(self):

        self.model = None

        if self.get_use_better_transformer() and _has_better_transformer:
            try:
                with init_empty_weights():
                    self.model = self._empty_model_from_config()
                    self.model = BetterTransformer.transform(self.model)
            except ValueError as ve:
                del self.model
                clean_memory()
                self.model = None

            if self.model is None:
                try:

                    print(f"new version of transfomer, no need to use BetterTransformer, try setting attn impl to sdpa...", file=stderr)
                    self.config.attn_implementation = "sdpa"

                    with init_empty_weights():
                        self.model = self._empty_model_from_config(attn_implementation="sdpa")
                    print(f"attn imp: {type(self.model.model.layers[3].self_attn)}", file=stderr)

                except TypeError as ve:
                    del self.model
                    clean_memory()
                    self.model = None

        if self.model is None:
            print(f"either BetterTransformer or attn_implementation='sdpa' is available, creating model directly", file=stderr)
            with init_empty_weights():
                self.model = self._empty_model_from_config()

        quantization_config = getattr(self.config, "quantization_config", None)

        if quantization_config is not None and self.moe_selective_fused_adapter_name != "gpt_oss_mxfp4_reference":
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model = self.model, device_map = device_map)

        self.model.eval()
        self.model.tie_weights()

        self.set_layers_from_layer_names()

        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer,
                                        dtype=self.running_dtype)

        if 'rotary_pos_emb' in self.layer_names_dict:
            self.load_rotary_pos_emb_to_device()

    def _empty_model_from_config(self, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        if getattr(self, "_qwen35_moe_layout", False):
            import transformers as transformers_module

            for auto_class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
                auto_class = getattr(transformers_module, auto_class_name, None)
                if auto_class is None:
                    continue
                try:
                    return auto_class.from_config(self.config, **kwargs)
                except Exception:
                    continue
        return AutoModelForCausalLM.from_config(self.config, **kwargs)

    def set_layers_from_layer_names(self):

        self.layers = []

        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    def load_rotary_pos_emb_to_device(self):
        state_dict = load_layer(self.checkpoint_path, self.layer_names_dict['rotary_pos_emb'])
        self.move_layer_to_device(state_dict)

    def load_layer_to_cpu(self, layer_name):
        load_started = time.perf_counter()
        if self._lazy_split_builder is not None:
            if ".experts." in layer_name:
                if self.expert_materialization_mode != "direct_slice":
                    layer, expert_id, _ = self._lazy_split_builder._parse_expert_shard_name(layer_name)
                    self._lazy_split_builder.ensure_expert_ready(layer, expert_id)
            else:
                self._lazy_split_builder.ensure_layer_dense_ready(layer_name)

        t = time.time()

        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time

            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)

            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        self._runtime_stats['cpu_load_calls'] += 1
        self._runtime_stats['cpu_load_bytes'] += self._state_dict_nbytes(state_dict)
        load_elapsed = time.perf_counter() - load_started
        self._runtime_stats['cpu_load_seconds'] += load_elapsed
        self._layer_runtime_stats(layer_name)['dense_load_seconds'] += load_elapsed

        if self.prefetching:
            t = time.time()
            if torch.cuda.is_available():
                for k in state_dict.keys():
                    state_dict[k].pin_memory()
            else:
                print("Prefetching is enabled, but no pin_memory operation is needed for CPU.", file=stderr)

            elapsed_time = time.time() - t
            if self.profiling_mode:
                self.profiler.add_profiling_time('pin_memory_to_trigger_load', elapsed_time)

        return state_dict

    def move_layer_to_device(self, state_dict):
        move_started = time.perf_counter()
        self._runtime_stats['device_load_calls'] += 1
        self._runtime_stats['device_load_bytes'] += self._state_dict_nbytes(state_dict)

        layers = []
        for param_name, param in state_dict.items():
            if self.hf_quantizer is None:
                layers.append(param_name)
            else:
                if '.weight' in param_name:
                    layer_name = param_name[:param_name.index(".weight") + len(".weight")]
                    if layer_name not in layers:
                        layers.append(layer_name)

        layer_name_for_stats = self._layer_name_from_param(layers[0]) if layers else None
        for param_name in layers:
            if (self.hf_quantizer is None or
                not self.hf_quantizer.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})
               ):
                set_module_tensor_to_device(self.model, param_name, self.running_device, value=state_dict[param_name],
                                            dtype=self.running_dtype,
                                            )
            else:
                torch_dtype = self.hf_quantizer.update_torch_dtype(None)
                self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name, self.running_device, state_dict)
        move_elapsed = time.perf_counter() - move_started
        self._runtime_stats['device_load_seconds'] += move_elapsed
        if layer_name_for_stats is not None:
            self._layer_runtime_stats(layer_name_for_stats)['move_to_device_seconds'] += move_elapsed
        self._check_cuda_vram_budget(f"move_layer_to_device:{layer_name_for_stats}")
        return layers

    @staticmethod
    def _state_dict_nbytes(state_dict):
        total = 0
        for tensor in state_dict.values():
            if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                total += tensor.numel() * tensor.element_size()
        return total

    def reset_runtime_stats(self):
        self._runtime_stats = {
            'cpu_load_calls': 0,
            'cpu_load_bytes': 0,
            'cpu_load_seconds': 0.0,
            'device_load_calls': 0,
            'device_load_bytes': 0,
            'device_load_seconds': 0.0,
            'peak_cuda_allocated_bytes': 0,
            'peak_cuda_reserved_bytes': 0,
            'prefetch_wait_seconds': 0.0,
            'layer_forward_seconds': 0.0,
            'position_args_seconds': 0.0,
            'offload_seconds': 0.0,
            'cleanup_seconds': 0.0,
            'layer_loop_seconds': 0.0,
            'layer_runtime_breakdown': {},
        }

    def runtime_stats(self):
        if torch.cuda.is_available():
            self._runtime_stats['peak_cuda_allocated_bytes'] = max(
                int(self._runtime_stats.get('peak_cuda_allocated_bytes', 0)),
                int(torch.cuda.max_memory_allocated()),
            )
            self._runtime_stats['peak_cuda_reserved_bytes'] = max(
                int(self._runtime_stats.get('peak_cuda_reserved_bytes', 0)),
                int(torch.cuda.max_memory_reserved()),
            )
        return dict(self._runtime_stats)

    def _check_cuda_vram_budget(self, stage):
        if self.max_vram_bytes is None or not torch.cuda.is_available():
            return
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        self._runtime_stats['peak_cuda_allocated_bytes'] = max(
            int(self._runtime_stats.get('peak_cuda_allocated_bytes', 0)),
            int(torch.cuda.max_memory_allocated()),
        )
        self._runtime_stats['peak_cuda_reserved_bytes'] = max(
            int(self._runtime_stats.get('peak_cuda_reserved_bytes', 0)),
            int(torch.cuda.max_memory_reserved()),
        )
        if max(allocated, reserved) > self.max_vram_bytes:
            raise MemoryError(
                f"CUDA VRAM budget exceeded at {stage}: "
                f"allocated={allocated / 1024 ** 2:.1f}MB reserved={reserved / 1024 ** 2:.1f}MB "
                f"budget={self.max_vram_bytes / 1024 ** 2:.1f}MB"
            )

    def _layer_runtime_stats(self, layer_name):
        return self._runtime_stats['layer_runtime_breakdown'].setdefault(
            str(layer_name),
            {
                'dense_load_seconds': 0.0,
                'prefetch_wait_seconds': 0.0,
                'move_to_device_seconds': 0.0,
                'position_args_seconds': 0.0,
                'layer_forward_seconds': 0.0,
                'offload_seconds': 0.0,
                'cleanup_seconds': 0.0,
                'layer_loop_seconds': 0.0,
            },
        )

    def _layer_name_from_param(self, param_name):
        param_name = str(param_name)
        for layer_name in getattr(self, 'layer_names', []):
            if param_name == layer_name or param_name.startswith(f"{layer_name}."):
                return layer_name
        return param_name.rsplit('.', 1)[0]

    def _should_keep_layer_resident(self, layer_name):
        return (
            self.keep_nontransformer_resident
            and layer_name in getattr(self, '_nontransformer_resident_layer_names', set())
        )

    def _next_streamed_layer_name(self, start_idx):
        for layer_name in self.layer_names[start_idx:]:
            if layer_name not in self._resident_layer_names:
                return layer_name
        return None

    def _prepare_resident_nontransformer_layers(self):
        for layer_name in self.layer_names:
            if not self._should_keep_layer_resident(layer_name):
                continue
            state_dict = self.load_layer_to_cpu(layer_name)
            self.move_layer_to_device(state_dict)
            self._resident_layer_names.add(layer_name)

    def check_kv_cache_safety(self):
        estimate = self.estimate_kv_cache_bytes()
        if estimate is None:
            return

        available = self._available_ram_bytes()
        if available is None:
            return

        reserved = int(self.os_reserved_ram_mb) * 1024 * 1024
        safe_available = max(0, available - reserved)
        if estimate <= safe_available:
            return

        message = (
            f"WARNING: estimated KV cache for ctx={self.max_seq_len} is "
            f"{estimate / 1024 ** 3:.2f}GB, but available RAM after reserving "
            f"{self.os_reserved_ram_mb}MB for the OS is {safe_available / 1024 ** 3:.2f}GB. "
            "Long context may make the system unstable."
        )
        if self.abort_unsafe_context:
            raise MemoryError(message)
        print(message, file=stderr)

    def estimate_kv_cache_bytes(self):
        hidden_size = getattr(self.config, "hidden_size", None)
        num_layers = getattr(self.config, "num_hidden_layers", None) or getattr(self.config, "n_layer", None)
        num_heads = getattr(self.config, "num_attention_heads", None) or getattr(self.config, "n_head", None)
        kv_heads = getattr(self.config, "num_key_value_heads", None) or num_heads
        if not hidden_size or not num_layers or not num_heads or not kv_heads:
            return None

        head_dim = getattr(self.config, "head_dim", None) or hidden_size // num_heads
        dtype_bytes = torch.tensor([], dtype=self.running_dtype).element_size()
        return int(num_layers) * 2 * int(self.max_seq_len) * int(kv_heads) * int(head_dim) * dtype_bytes

    @staticmethod
    def _available_ram_bytes():
        try:
            import psutil
            return psutil.virtual_memory().available
        except Exception:
            pass

        if platform == "win32":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = MEMORYSTATUSEX()
                status.dwLength = ctypes.sizeof(status)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                return int(status.ullAvailPhys)
            except Exception:
                return None
        return None

    def offload_layer_from_device(self, layer, moved_layers):
        if self.hf_quantizer is not None:
            for param_name in moved_layers:
                set_module_tensor_to_device(self.model, param_name, 'meta')
        else:
            layer.to("meta")

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = self.get_past_key_values_cache_seq_len(past_key_values)

            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_past_key_values_cache_seq_len(self, past_key_values):
        if self._is_cache_object(past_key_values):
            return int(past_key_values.get_seq_length())
        return past_key_values[0][0].shape[2]
    def get_sequence_len(self, seq):
        return seq.shape[1]

    def get_pos_emb_args(self, len_p, len_s):
        if getattr(self.config, "model_type", None) == "gpt_oss":
            rotary_emb = getattr(getattr(self.model, "model", None), "rotary_emb", None)
            if rotary_emb is None:
                return {}
            position_ids = torch.arange(
                len_p,
                len_p + len_s,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)
            dummy_hidden = torch.empty(
                (1, len_s, getattr(self.config, "hidden_size")),
                device=self.device,
                dtype=self.running_dtype,
            )
            return {"position_embeddings": rotary_emb(dummy_hidden, position_ids)}
        if self._is_qwen35_config():
            rotary_emb = self._qwen35_rotary_emb()
            text_config = self._qwen35_text_config()
            hidden_size = getattr(text_config, "hidden_size", getattr(self.config, "hidden_size", None))
            if rotary_emb is None or hidden_size is None:
                return {}
            base_position_ids = torch.arange(
                len_p,
                len_p + len_s,
                device=self.device,
                dtype=torch.long,
            )
            position_ids = base_position_ids.view(1, 1, len_s).expand(3, 1, len_s)
            dummy_hidden = torch.empty(
                (1, len_s, int(hidden_size)),
                device=self.device,
                dtype=self.running_dtype,
            )
            return {"position_embeddings": rotary_emb(dummy_hidden, position_ids)}
        return {}

    @staticmethod
    def _first_decoder_output(layer_output):
        if isinstance(layer_output, (tuple, list)):
            return layer_output[0]
        return layer_output

    def _supports_transformers_cache(self):
        return bool(cache_utils_installed and (getattr(self.config, "model_type", None) == "gpt_oss" or self._is_qwen35_config()))

    def _is_cache_object(self, past_key_values):
        return bool(cache_utils_installed and isinstance(past_key_values, Cache))

    def _new_transformers_cache(self):
        cache_config = self._qwen35_text_config() if self._is_qwen35_config() else self.config
        try:
            return DynamicCache(config=cache_config)
        except TypeError:
            return DynamicCache()

    def get_past_key_value_args(self, k_cache=None, v_cache=None, past_key_values=None):
        if self._is_cache_object(past_key_values):
            return {'past_key_values': past_key_values}
        return {'past_key_value': (k_cache, v_cache)}

    def _gpt_oss_layer_type(self, layer_name):
        if getattr(self.config, "model_type", None) != "gpt_oss":
            return None
        prefix = f"{self.layer_names_dict['layer_prefix']}."
        if not str(layer_name).startswith(prefix):
            return None
        try:
            layer_idx = int(str(layer_name)[len(prefix):].split(".", 1)[0])
        except (TypeError, ValueError):
            return None
        layer_types = getattr(self.config, "layer_types", None) or []
        if 0 <= layer_idx < len(layer_types):
            return layer_types[layer_idx]
        return None

    def _qwen35_text_config(self):
        return getattr(self.config, "text_config", self.config)

    def _is_qwen35_config(self):
        model_type = str(getattr(self.config, "model_type", "") or "").lower()
        text_model_type = str(getattr(self._qwen35_text_config(), "model_type", "") or "").lower()
        return model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} or text_model_type == "qwen3_5_moe_text"

    def _qwen35_rotary_emb(self):
        model_root = getattr(self.model, "model", None)
        language_model = getattr(model_root, "language_model", None)
        for candidate in (language_model, model_root):
            rotary_emb = getattr(candidate, "rotary_emb", None)
            if rotary_emb is not None:
                return rotary_emb
        return None

    def _qwen35_layer_type(self, layer_name):
        if not self._is_qwen35_config():
            return None
        prefix = f"{self.layer_names_dict['layer_prefix']}."
        if not str(layer_name).startswith(prefix):
            return None
        try:
            layer_idx = int(str(layer_name)[len(prefix):].split(".", 1)[0])
        except (TypeError, ValueError):
            return None
        layer_types = getattr(self._qwen35_text_config(), "layer_types", None) or []
        if 0 <= layer_idx < len(layer_types):
            return layer_types[layer_idx]
        return None

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s, layer_name=None):
        if self._qwen35_layer_type(layer_name) == "linear_attention":
            return {'attention_mask': None}
        key_width = len_p + len_s
        if self._gpt_oss_layer_type(layer_name) == "sliding_attention":
            sliding_window = int(getattr(self.config, "sliding_window", 0) or 0)
            if sliding_window > 0:
                key_width = min(key_width, sliding_window)
        mask = full_attention_mask[:, :, -len_s:, -key_width:]
        if self._is_qwen35_config():
            dtype = self.running_dtype if self.running_dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.float32
            additive_mask = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
            additive_mask = additive_mask.masked_fill(~mask, torch.finfo(dtype).min)
            return {'attention_mask': additive_mask}
        return {'attention_mask': mask}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):

        return {'position_ids': full_position_ids[:, len_p:len_p + len_s]}


    def run_lm_head(self, layer, seq):
        return layer(seq).float()

    def run_norm(self, layer, seq):
        return layer(seq)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        use_cache = bool(use_cache)
        supports_transformers_cache = self._supports_transformers_cache()
        if cache_utils_installed and use_cache and not supports_transformers_cache:
            use_cache = False
        if use_cache and supports_transformers_cache and past_key_values is None:
            past_key_values = self._new_transformers_cache()
        cache_is_object = self._is_cache_object(past_key_values)

        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()

        if self.reinitialize_each_forward:
            del self.model
            clean_memory()
            self.init_model()

        using_inputs_embeds = inputs_embeds is not None
        if using_inputs_embeds:
            batch = [inputs_embeds_unit.to(self.running_device).unsqueeze(0) for inputs_embeds_unit in inputs_embeds]
        else:
            batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])

        attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]
        cache_past_length = (
            self.get_past_key_values_cache_seq_len(past_key_values)
            if past_key_values is not None
            else 0
        )

        kv_cache_list = past_key_values if use_cache and cache_is_object else ([] if use_cache else None)
        if use_cache and not cache_is_object:
            for x in self.layers:
                kv_cache_list.append(([], []))
        all_hidden_states = [] * len(self.layers) if output_hidden_states else None
        all_self_attns = [] * len(self.layers) if output_attentions else None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            if self.prefetching:
                first_streamed_layer = self._next_streamed_layer_name(0)
                future = (
                    executor.submit(self.load_layer_to_cpu, first_streamed_layer)
                    if first_streamed_layer is not None
                    else None
                )


            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                               desc=f'running layers({self.running_device})',
                                               total=len(self.layers),
                                               disable=self.quiet_progress):
                layer_loop_started = time.perf_counter()
                layer_stats = self._layer_runtime_stats(layer_name)
                layer_is_resident = layer_name in self._resident_layer_names
                moved_layers = []

                if layer_is_resident:
                    pass
                elif self.prefetching:
                    if self.profiling_mode:
                        t = time.time()
                    prefetch_wait_started = time.perf_counter()
                    state_dict = future.result() if future is not None else self.load_layer_to_cpu(layer_name)
                    prefetch_wait_elapsed = time.perf_counter() - prefetch_wait_started
                    self._runtime_stats['prefetch_wait_seconds'] += prefetch_wait_elapsed
                    layer_stats['prefetch_wait_seconds'] += prefetch_wait_elapsed
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('load_safe_tensor_cpu_wait', elapsed_time)

                    if self.profiling_mode:
                        t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_state_dict', elapsed_time)

                    next_streamed_layer = self._next_streamed_layer_name(i + 1)
                    if next_streamed_layer is not None:

                        if self.profiling_mode:
                            t = time.time()
                        future = executor.submit(self.load_layer_to_cpu, next_streamed_layer)

                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('kick_off_load_cpu', elapsed_time)
                    else:
                        future = None

                else:
                    state_dict = self.load_layer_to_cpu(layer_name)
                    if self.profiling_mode:
                        t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_safe_tensor', elapsed_time)

                for j, seq in enumerate(batch):

                    if layer_name == self.layer_names_dict['embed']:
                        layer_forward_started = time.perf_counter()
                        if not using_inputs_embeds:
                            batch[j] = layer(seq)
                        elapsed = time.perf_counter() - layer_forward_started
                        self._runtime_stats['layer_forward_seconds'] += elapsed
                        layer_stats['layer_forward_seconds'] += elapsed
                    elif layer_name == self.layer_names_dict['norm']:

                        layer_forward_started = time.perf_counter()
                        batch[j] = self.run_norm(layer, seq)
                        elapsed = time.perf_counter() - layer_forward_started
                        self._runtime_stats['layer_forward_seconds'] += elapsed
                        layer_stats['layer_forward_seconds'] += elapsed

                        if output_attentions:
                            all_hidden_states[i].append(batch[j])
                    elif layer_name == self.layer_names_dict['lm_head']:
                        layer_forward_started = time.perf_counter()
                        batch[j] = self.run_lm_head(layer, seq)
                        elapsed = time.perf_counter() - layer_forward_started
                        self._runtime_stats['layer_forward_seconds'] += elapsed
                        layer_stats['layer_forward_seconds'] += elapsed
                    else:

                        if output_attentions:
                            all_hidden_states[i].append(new_seq)

                        if past_key_values is not None:
                            if cache_is_object:
                                k_cache, v_cache = None, None
                            else:
                                k_cache, v_cache = past_key_values[i - 1]
                            len_p = cache_past_length
                            len_s = self.get_sequence_len(seq)

                            position_started = time.perf_counter()
                            position_ids_args = self.get_position_ids_args(position_ids, len_p, len_s)
                            attention_mask_args = self.get_attention_mask_args(attention_mask, len_p, len_s, layer_name=layer_name)
                            past_key_value_args = self.get_past_key_value_args(
                                k_cache,
                                v_cache,
                                past_key_values=past_key_values if cache_is_object else None,
                            )

                            kwargs = {'use_cache':True,
                                      }

                            pos_embed_args = self.get_pos_emb_args(len_p, len_s)
                            kwargs = {**kwargs, **past_key_value_args, **pos_embed_args, **attention_mask_args,
                                      **position_ids_args}
                            position_elapsed = time.perf_counter() - position_started
                            self._runtime_stats['position_args_seconds'] += position_elapsed
                            layer_stats['position_args_seconds'] += position_elapsed


                            layer_forward_started = time.perf_counter()
                            layer_outputs = layer(seq,
                                                  **kwargs
                                                  )
                            elapsed = time.perf_counter() - layer_forward_started
                            self._runtime_stats['layer_forward_seconds'] += elapsed
                            layer_stats['layer_forward_seconds'] += elapsed
                            new_seq = self._first_decoder_output(layer_outputs)

                            if output_attentions:
                                all_self_attns[i].append(layer_outputs[1])

                            if use_cache and not cache_is_object:
                                (k_cache, v_cache) = layer_outputs[2 if output_attentions else 1]
                                kv_cache_list[i][0].append(k_cache)
                                kv_cache_list[i][1].append(v_cache)


                        else:
                            len_seq = self.get_sequence_len(seq)


                            position_started = time.perf_counter()
                            pos_embed_args = self.get_pos_emb_args(0, len_seq)
                            attention_mask_args = self.get_attention_mask_args(attention_mask, 0, len_seq, layer_name=layer_name)
                            position_ids_args = self.get_position_ids_args(position_ids, 0, len_seq)
                            position_elapsed = time.perf_counter() - position_started
                            self._runtime_stats['position_args_seconds'] += position_elapsed
                            layer_stats['position_args_seconds'] += position_elapsed


                            if not use_cache:

                                kwargs = {'use_cache': False,
                                          'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                                          }
                                kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}


                                layer_forward_started = time.perf_counter()
                                new_seq = self._first_decoder_output(layer(seq, **kwargs))
                                elapsed = time.perf_counter() - layer_forward_started
                                self._runtime_stats['layer_forward_seconds'] += elapsed
                                layer_stats['layer_forward_seconds'] += elapsed
                            else:

                                kwargs = {'use_cache': True,
                                          'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                                          }
                                if cache_is_object:
                                    past_key_value_args = self.get_past_key_value_args(past_key_values=past_key_values)
                                    kwargs = {**kwargs, **past_key_value_args}
                                kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}

                                layer_forward_started = time.perf_counter()
                                layer_out = layer(seq, **kwargs)
                                elapsed = time.perf_counter() - layer_forward_started
                                self._runtime_stats['layer_forward_seconds'] += elapsed
                                layer_stats['layer_forward_seconds'] += elapsed

                                if cache_is_object:
                                    new_seq = self._first_decoder_output(layer_out)
                                else:
                                    new_seq, (k_cache, v_cache) = layer_out
                                    kv_cache_list[i][0].append(k_cache)
                                    kv_cache_list[i][1].append(v_cache)


                        batch[j] = new_seq

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                offload_started = time.perf_counter()
                if self._should_keep_layer_resident(layer_name):
                    self._resident_layer_names.add(layer_name)
                else:
                    self.offload_layer_from_device(layer, moved_layers)
                offload_elapsed = time.perf_counter() - offload_started
                self._runtime_stats['offload_seconds'] += offload_elapsed
                layer_stats['offload_seconds'] += offload_elapsed

                should_clean = self.layer_cleanup_interval == 1 or (
                    self.layer_cleanup_interval > 1 and (i + 1) % self.layer_cleanup_interval == 0
                )
                if should_clean:
                    cleanup_started = time.perf_counter()
                    clean_memory()
                    cleanup_elapsed = time.perf_counter() - cleanup_started
                    self._runtime_stats['cleanup_seconds'] += cleanup_elapsed
                    layer_stats['cleanup_seconds'] += cleanup_elapsed

                layer_loop_elapsed = time.perf_counter() - layer_loop_started
                self._runtime_stats['layer_loop_seconds'] += layer_loop_elapsed
                layer_stats['layer_loop_seconds'] += layer_loop_elapsed

        logits = torch.cat(batch, 0)
        past_key_values_to_return = None
        if use_cache:
            if cache_is_object:
                past_key_values_to_return = kv_cache_list
            else:
                kv_cache_list = kv_cache_list[1:-2]
                for i in range(len(kv_cache_list)):

                    kv_cache_list[i] = (torch.cat(kv_cache_list[i][0], 0), torch.cat(kv_cache_list[i][1], 0))
                past_key_values_to_return = tuple(kv_cache_list)


        if output_attentions:
            all_self_attns = all_self_attns[0:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states[0:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)

        if not return_dict:
            return tuple(v for v in [logits,
                                     past_key_values_to_return,
                                     tuple(all_hidden_states) if all_hidden_states is not None else None,
                                     tuple(all_self_attns) if all_self_attns is not None else None] if v is not None)
        if self.profiling_mode:
            forward_elapsed_time = time.process_time() - forward_start
            forward_elapsed_time_wall = time.time() - forward_start_wall
            self.profiler.print_profiling_time()


            print(f"total infer process time(including all above plus gpu compute): {forward_elapsed_time:.04f}", file=stderr)
            print(f"total infer wall time(including all above plus gpu compute): {forward_elapsed_time_wall:.04f}", file=stderr)

            self.profiler.clear_profiling_time()


        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values_to_return,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_hidden_states is not None else None,
        )
