import gc
import json
import os
import ctypes
import shutil
import re
from tqdm import tqdm
from pathlib import Path
from glob import glob
import time

from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
from sys import platform, stderr

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True


import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .persist import ModelPersister


try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
except ImportError:
    bitsandbytes_installed = False


import huggingface_hub


# replacement for bnb quantstat.as_dict(True), until the bug is fixed....
def save_quant_state_to_dict(self, packed=True):
    """
    returns dict of tensors and strings to use in serialization via _save_to_state_dict()
    param: packed -- returns dict[str, torch.Tensor] for state_dict
    """
    qs_dict = {
        'quant_type': self.quant_type,
        'absmax': self.absmax,
        'blocksize': self.blocksize,
        'quant_map': self.code,
        'dtype': str(self.dtype).strip('torch.'),
        'shape': tuple(self.shape),
    }
    if self.nested:
        qs_dict.update({
            'nested_absmax': self.state2.absmax,
            'nested_blocksize': self.state2.blocksize,
            'nested_quant_map': self.state2.code,
            'nested_dtype': str(self.state2.dtype).strip('torch.'),
            'nested_offset': self.offset.item(),
        })
    if not packed:
        return qs_dict

    qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
    non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
    qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = bnb.utils.pack_dict_to_tensor(non_tensor_dict)
    return qs_packed_dict



class NotEnoughSpaceException(Exception):
    pass

# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as ex:
        # maybe platform
        pass
    torch.cuda.empty_cache()


def uncompress_layer_state_dict(layer_state_dict):
    uncompressed_layer_state_dict = None
    if any(['4bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '4bit' not in k:
                quant_state_dict = {kk[len(k):]: kv for kk, kv in layer_state_dict.items() if kk.startswith(k) and k != kk}
                quant_state = bnb.functional.QuantState.from_dict(qs_dict=quant_state_dict, device="cuda")

                dqv = bnb.functional.dequantize_nf4(v.cuda(), quant_state)
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict
    elif any(['8bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '8bit' not in k:

                absmax = layer_state_dict[k + ".8bit.absmax"]
                code = layer_state_dict[k + ".8bit.code"]

                dqv = bnb.functional.dequantize_blockwise(v.cuda(),
                                                          bnb.functional.QuantState(absmax=absmax.cuda(),
                                                                                    code=code.cuda(),
                                                                                    blocksize=2048,
                                                                                    dtype=torch.float16))
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict

    return layer_state_dict if uncompressed_layer_state_dict is None else uncompressed_layer_state_dict

def load_layer(local_path, layer_name, profiling=False):
    #layer_state_dict = load_file(Path(local_path) / (layer_name + ".safetensors"), device="cpu")
    layer_state_dict = ModelPersister.get_model_persister().load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    to_return = uncompress_layer_state_dict(layer_state_dict)

    #clean_memory()

    if profiling:
        elapsed_time = time.process_time() - t
        return to_return, elapsed_time
    else:
        return to_return



def check_space(checkpoint_path, layer_shards_saving_path=None, compression=None, splitted_model_dir_name='splitted_model'):
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / '*')):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == '4bit':
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    total, used, free = shutil.disk_usage(checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path)

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(f"Not enough space. Free space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path}:"  \
                                      f" {free / 1024 / 1024 / 1024:.02f}GB. Model total size: {total_shard_files_size_bytes / 1024 / 1024 / 1024:.02f}GB. " \
                                      f"existing space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path} assuming can reuse: {total_saved_split_files_size_bytes/ 1024 / 1024 / 1024:.02f}GB. "
                                      )


def build_safetensors_index_from_local_shards(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    shard_paths = sorted(checkpoint_path.glob("*.safetensors"))
    if not shard_paths:
        return None

    weight_map = {}
    total_size = 0
    for shard_path in shard_paths:
        total_size += int(shard_path.stat().st_size)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(f"duplicate tensor key {key!r} in local safetensors shards")
                weight_map[key] = shard_path.name

    return {
        "metadata": {
            "total_size": total_size,
            "index_source": "local_safetensors_scan",
        },
        "weight_map": weight_map,
    }


def load_checkpoint_weight_map(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if os.path.exists(checkpoint_path / 'pytorch_model.bin.index.json'):
        with open(checkpoint_path / 'pytorch_model.bin.index.json', 'rb') as f:
            return json.load(f)['weight_map'], False
    if os.path.exists(checkpoint_path / 'model.safetensors.index.json'):
        with open(checkpoint_path / 'model.safetensors.index.json', 'rb') as f:
            return json.load(f)['weight_map'], True
    scanned = build_safetensors_index_from_local_shards(checkpoint_path)
    if scanned:
        return scanned['weight_map'], True
    raise FileNotFoundError("Expected model.safetensors.index.json, pytorch_model.bin.index.json, or local *.safetensors shards")

def compress_layer_state_dict(layer_state_dict, compression=None):
    compressed_layer_state_dict = None
    if compression == '4bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(v.cuda(), blocksize=64)
            compressed_layer_state_dict[k] = v_quant
            for quant_state_k, quant_state_v in save_quant_state_to_dict(quant_state).items():
                compressed_layer_state_dict[k + ".4bit." + quant_state_k] = quant_state_v
    elif compression == '8bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_blockwise(v.cuda(), blocksize=2048)
            absmax = quant_state.absmax.clone().contiguous()
            code = quant_state.code.clone().contiguous()
            compressed_layer_state_dict[k] = v_quant
            compressed_layer_state_dict[k + ".8bit.absmax"] = absmax
            compressed_layer_state_dict[k + ".8bit.code"] = code

    return compressed_layer_state_dict if compressed_layer_state_dict is not None else layer_state_dict


_EXPERT_PREFIX_RE = re.compile(r"^(?P<prefix>.+\.experts\.[^.]+)\.")


def _iter_config_scopes(config):
    yield config
    for nested_name in ("text_config", "language_config", "llm_config"):
        nested = config.get(nested_name) if isinstance(config, dict) else None
        if isinstance(nested, dict):
            yield nested


def _get_nested_config_value(config, names):
    for scope in _iter_config_scopes(config):
        value = _get_config_value(scope, names)
        if value is not None:
            return value
    return None


def _get_config_value(config, names):
    for name in names:
        if isinstance(config, dict) and name in config:
            return config[name]
    return None


def infer_num_experts_from_config(checkpoint_path):
    config_path = Path(checkpoint_path) / "config.json"
    if not os.path.exists(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    value = _get_nested_config_value(
        config,
        (
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "num_routed_experts",
            "n_experts",
            "moe_num_experts",
        ),
    )
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_fused_expert_axis(tensor, num_experts):
    if num_experts is None or not hasattr(tensor, "shape"):
        return None
    candidate_axes = [axis for axis, size in enumerate(tensor.shape) if int(size) == int(num_experts)]
    if not candidate_axes:
        return None
    return candidate_axes[0]


def _split_fused_expert_tensor(key, tensor, num_experts):
    expert_marker = ".experts."
    if expert_marker not in key:
        return None

    _, suffix = key.split(expert_marker, 1)
    first_suffix_part = suffix.split(".", 1)[0]
    if first_suffix_part.isdigit():
        return None

    expert_axis = _find_fused_expert_axis(tensor, num_experts)
    if expert_axis is None:
        return None

    prefix, suffix = key.split(expert_marker, 1)
    if "." in suffix:
        fused_name, remainder = suffix.split(".", 1)
    else:
        fused_name, remainder = suffix, "weight"

    split_tensors = {}
    for expert_idx in range(num_experts):
        expert_tensor = tensor.select(expert_axis, expert_idx).contiguous()
        expert_prefix = f"{prefix}.experts.{expert_idx}"
        expert_key = f"{expert_prefix}.{fused_name}.{remainder}"
        split_tensors.setdefault(expert_prefix, {})[expert_key] = expert_tensor
    return split_tensors


def split_moe_layer_state_dict(
        layer_state_dict,
        num_experts=None,
        return_metadata=False,
        drop_fused_from_dense=False):
    """Split a transformer block state dict into dense tensors and experts.

    The direct matcher keys off the common HuggingFace convention
    ``*.experts.<id>.*``. Fused tensors such as ``*.experts.gate_up_proj``
    are also split when ``num_experts`` can be inferred and one tensor
    dimension equals that expert count. Shared experts stay in the dense
    shard because they are active for every token.
    """
    dense_state_dict = {}
    expert_state_dicts = defaultdict(dict)
    metadata = {
        "direct_expert_prefixes": set(),
        "fused_expert_prefixes": set(),
        "fused_tensor_keys": set(),
    }

    for key, value in layer_state_dict.items():
        fused_splits = _split_fused_expert_tensor(key, value, num_experts)
        if fused_splits is not None:
            metadata["fused_tensor_keys"].add(key)
            for expert_prefix, expert_state_dict in fused_splits.items():
                expert_state_dicts[expert_prefix].update(expert_state_dict)
                metadata["fused_expert_prefixes"].add(expert_prefix)
            if not drop_fused_from_dense:
                dense_state_dict[key] = value
            continue

        match = _EXPERT_PREFIX_RE.match(key)
        if match is None:
            dense_state_dict[key] = value
            continue

        expert_prefix = match.group("prefix")
        metadata["direct_expert_prefixes"].add(expert_prefix)
        expert_state_dicts[expert_prefix][key] = value

    if return_metadata:
        metadata = {
            key: sorted(value)
            for key, value in metadata.items()
        }
        return dense_state_dict, dict(expert_state_dicts), metadata

    return dense_state_dict, dict(expert_state_dicts)

def remove_real_and_linked_file(to_delete):
    if (os.path.realpath(to_delete) != to_delete):
        targetpath = os.path.realpath(to_delete)

    os.remove(to_delete)
    if (targetpath):
         os.remove(targetpath)



def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None,
                          moe_expert_sharding=False, moe_strict_streaming=False,
                          moe_allow_dense_fallback=False, moe_selective_fused_adapter_name=None,
                          resume_split_build=False, lazy_build=False):
    """
    Save the all layers of a model sharded checkpoint using safetensors.
    """

    if compression is not None:
        assert bitsandbytes_installed, f"when using compression bitsandbytes has to be installed."
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    if moe_expert_sharding:
        splitted_model_dir_name = splitted_model_dir_name + ".moe"

    checkpoint_path = Path(checkpoint_path)
    moe_num_experts = infer_num_experts_from_config(checkpoint_path) if moe_expert_sharding else None


    saving_path = checkpoint_path / splitted_model_dir_name

    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name

    if moe_expert_sharding and (resume_split_build or lazy_build):
        from .moe_split_builder import IncrementalMoESplitBuilder

        builder = IncrementalMoESplitBuilder(
            checkpoint_path,
            saving_path,
            layer_names=layer_names,
            repo_id=repo_id,
            hf_token=hf_token,
            strict_mode=moe_strict_streaming,
            adapter_name=moe_selective_fused_adapter_name,
            allow_dense_fallback=moe_allow_dense_fallback,
        )
        if lazy_build:
            return str(saving_path)
        builder.build_until_ready()
        return str(saving_path)


    index, safetensors_format = load_checkpoint_weight_map(checkpoint_path)

    if layer_names is None:
        n_layers = len(set([int(k.split('.')[2]) for k in index.keys() if 'model.layers' in k]))
    else:
        n_layers = len(set([int(k[len(layer_names['layer_prefix']):].split('.')[1]) for k in index.keys() if layer_names['layer_prefix'] in k]))

    if layer_names is None:
        layers = ['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)] + ['model.norm.', 'lm_head.']
    else:
        layers = [layer_names['embed']] + [f'{layer_names["layer_prefix"]}.{i}' for i in range(n_layers)] + [layer_names['norm'], layer_names['lm_head']]

        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
        layers = [l + "." for l in layers]


    # check if splitting exists and all files are there
    found_layers = None
    #print(f"checking exists: {saving_path}")
    moe_expert_manifest_path = saving_path / "moe_expert_index.json"

    if os.path.exists(saving_path):
        # dir already exists, check if all layer files are there

        found_layers = {}
        for layer in layers:
            found_layers[layer] = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)

        print(f"found_layers:{found_layers}", file=stderr)
        if all(found_layers.values()):
            if moe_expert_sharding:
                if os.path.exists(moe_expert_manifest_path):
                    with open(moe_expert_manifest_path, "r", encoding="utf-8") as f:
                        moe_manifest = json.load(f)
                        expert_layer_names = moe_manifest.get("expert_layer_names", [])
                    if moe_manifest.get("has_fused_experts") and not moe_manifest.get("selective_fused_runtime"):
                        if moe_strict_streaming:
                            raise NotImplementedError(
                                "Existing MoE split contains fused experts without selective fused runtime; "
                                "strict streaming cannot use this split."
                            )
                        if not moe_allow_dense_fallback:
                            raise NotImplementedError(
                                "Existing MoE split requires dense fused fallback. "
                                "Pass moe_allow_dense_fallback=True to use it."
                            )
                    found_experts = [
                        ModelPersister.get_model_persister().model_persist_exist(expert_name + ".", saving_path)
                        for expert_name in expert_layer_names
                    ]
                    if all(found_experts):
                        print(f"saved MoE layers already found in {saving_path}", file=stderr)
                        return str(saving_path)
                print(f"MoE dense layer splits found, but expert manifest is missing or incomplete; re-saving layers.", file=stderr)
            else:
                # already downloaded, return saving path...
                print(f"saved layers already found in {saving_path}", file=stderr)
                return str(saving_path)
        else:
            print(f"some layer splits found, some are not, re-save all layers in case there's some corruptions.", file=stderr)

    if not delete_original:
        check_space(checkpoint_path, layer_shards_saving_path, compression, splitted_model_dir_name=splitted_model_dir_name)


    loaded_shard_files = set()
    state_dict = {}
    moe_expert_layer_names = set()
    moe_direct_expert_layer_names = set()
    moe_fused_expert_layer_names = set()
    moe_fused_tensor_names = set()
    dense_contains_full_fused_experts = False


    if not os.path.exists(saving_path):
        #os.makedirs(saving_path)
        saving_path.mkdir(parents=True, exist_ok=True)

    single_modelfile = None

    for layer_idx, layer in enumerate(tqdm(layers)):

        # Optionnally load next shard
        # checking whether after spliting from '-', if second element exists. otherwise it throws errors for single 'model.safetensor' files
        shard_files = sorted({v for k, v in index.items() if k.startswith(layer) and '-' in v and len(v.split('-')) > 1})
        if len(shard_files) > 0:
            for shard_file in shard_files:
                if shard_file in loaded_shard_files:
                    continue
                print(f'Loading shard {shard_file}', file=stderr)
                to_load = checkpoint_path / shard_file

                # check if to_load exist, if not downloaad it...
                if not os.path.exists(to_load):
                    assert repo_id is not None
                    huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                      token=hf_token)

                if not safetensors_format:
                    state_dict.update(torch.load(to_load, map_location='cpu'))
                else:
                    state_dict.update(load_file(to_load, device='cpu'))
                loaded_shard_files.add(shard_file)

        else:
            shards = [v for k, v in index.items() if k.startswith(layer)]
            single_modelfile = shards[0]
            to_load = checkpoint_path / single_modelfile
            # check if to_load exist, if not downloaad it...
            if not os.path.exists(to_load):
                assert repo_id is not None
                huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                token=hf_token)
            if not safetensors_format:
                state_dict.update(torch.load(to_load, map_location='cpu'))
            else:
                state_dict.update(load_file(to_load, device='cpu'))

        # Get layer state dict
        layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(layer)])
        original_layer_keys = list(layer_state_dict.keys())

        expert_state_dicts = {}
        if moe_expert_sharding:
            layer_state_dict, expert_state_dicts, moe_metadata = split_moe_layer_state_dict(
                layer_state_dict,
                num_experts=moe_num_experts,
                return_metadata=True,
                drop_fused_from_dense=bool(moe_selective_fused_adapter_name),
            )
            moe_direct_expert_layer_names.update(moe_metadata["direct_expert_prefixes"])
            moe_fused_expert_layer_names.update(moe_metadata["fused_expert_prefixes"])
            moe_fused_tensor_names.update(moe_metadata["fused_tensor_keys"])
            if moe_metadata["fused_tensor_keys"] and not moe_selective_fused_adapter_name:
                dense_contains_full_fused_experts = True
                if moe_strict_streaming:
                    raise NotImplementedError(
                        "Strict MoE streaming found fused expert tensors, but no selective fused adapter is registered. "
                        "Pass a real adapter name or disable strict mode."
                    )
                if not moe_allow_dense_fallback:
                    raise NotImplementedError(
                        "Fused expert tensors require dense fallback unless a selective adapter is registered. "
                        "Pass moe_allow_dense_fallback=True to keep full fused tensors in dense shards."
                    )

        layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

        # Save layer state dict as using safetensors

        marker_exists = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)
        if not marker_exists:
            ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

        for expert_prefix, expert_state_dict in expert_state_dicts.items():
            moe_expert_layer_names.add(expert_prefix)
            expert_layer_name = expert_prefix + "."
            expert_state_dict = compress_layer_state_dict(expert_state_dict, compression)
            marker_exists = ModelPersister.get_model_persister().model_persist_exist(expert_layer_name, saving_path)
            if not marker_exists:
                ModelPersister.get_model_persister().persist_model(expert_state_dict, expert_layer_name, saving_path)

        # Free memory
        for k in original_layer_keys:
            if k in state_dict:
                del state_dict[k]
        if layer_idx + 1 < len(layers):
            next_layer = layers[layer_idx + 1]
            next_shard_files = {v for k, v in index.items() if k.startswith(next_layer)}
        else:
            next_shard_files = set()
        for k in list(state_dict.keys()):
            if index.get(k) not in next_shard_files:
                del state_dict[k]
        loaded_shard_files = {v for v in loaded_shard_files if v in next_shard_files}
        del layer_state_dict
        del expert_state_dicts
        clean_memory()

    # deleting single modelfile if only a single modelfile was existing in hf repo 
    # and deletion of single modelfile should happen in the end if delete_original=True
    if delete_original and single_modelfile != None:
        to_delete = checkpoint_path / single_modelfile
        print(f"deleting original file: {to_delete}", file=stderr)
        remove_real_and_linked_file(to_delete)

    if moe_expert_sharding:
        has_fused_experts = bool(moe_fused_tensor_names)
        selective_fused_runtime = bool(has_fused_experts and moe_selective_fused_adapter_name)
        requires_dense_fallback = bool(has_fused_experts and not selective_fused_runtime)
        if moe_strict_streaming and has_fused_experts and not selective_fused_runtime:
            raise NotImplementedError("Strict MoE streaming could not enable selective fused runtime.")
        if not moe_expert_layer_names:
            print("WARNING: MoE expert sharding was enabled, but no '*.experts.<id>.*' tensors were found. "
                  "The model will fall back to dense layer streaming for those blocks.", file=stderr)
        with open(moe_expert_manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "moe_layout": "mixed" if moe_direct_expert_layer_names and has_fused_experts else (
                    moe_selective_fused_adapter_name if has_fused_experts and moe_selective_fused_adapter_name == "qwen3_5_moe"
                    else ("fused" if has_fused_experts else ("module_experts" if moe_direct_expert_layer_names else "unknown"))
                ),
                "has_fused_experts": has_fused_experts,
                "fused_experts_split": bool(moe_fused_expert_layer_names),
                "expert_layer_names": sorted(moe_expert_layer_names),
                "expert_count": len(moe_expert_layer_names),
                "direct_expert_layer_names": sorted(moe_direct_expert_layer_names),
                "direct_expert_count": len(moe_direct_expert_layer_names),
                "fused_expert_layer_names": sorted(moe_fused_expert_layer_names),
                "fused_expert_count": len(moe_fused_expert_layer_names),
                "fused_tensor_names": sorted(moe_fused_tensor_names),
                "fused_tensor_count": len(moe_fused_tensor_names),
                "num_experts_from_config": moe_num_experts,
                "selective_fused_runtime": selective_fused_runtime,
                "dense_contains_full_fused_experts": dense_contains_full_fused_experts,
                "requires_dense_fallback": requires_dense_fallback,
                "adapter_name": moe_selective_fused_adapter_name,
                "mxfp4_execution": (
                    "reference_dequant" if moe_selective_fused_adapter_name == "gpt_oss_mxfp4_reference" else None
                ),
                "shared_expert_remains_dense": moe_selective_fused_adapter_name == "qwen3_5_moe",
                "selective_fused_runtime_reason": (
                    "Selective fused adapter is registered."
                    if selective_fused_runtime else
                    "Fused tensors require dense fallback until a model-specific router adapter is registered."
                ),
            }, f, indent=2)

    return str(saving_path)

def find_or_create_local_splitted_path(model_local_path_or_repo_id, layer_shards_saving_path=None, compression=None,
                                       layer_names=None, hf_token=None, delete_original=False,
                                       moe_expert_sharding=False, moe_strict_streaming=False,
                                       moe_allow_dense_fallback=False, moe_selective_fused_adapter_name=None,
                                       resume_split_build=False, lazy_build=False):
    """
    find the model's local cache path, download the cache if not exists, then split and save the model.

    Parameters
    ----------
    model_local_path_or_repo_id : str
        model local path or hf repo id
    layer_shards_saving_path : str, optional
        optional path to save the splitted model, by default directly under the model local path

    Returns
    -------
    model_local_path : str
        local model path
    saved_layer_shards_path : str
        the path saved layer shards
    compression: str, optinal
        setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
    hf_token: str, optional
        huggingface api token could be provided, by default None
    """

    # try local model path, if the model exist split and save there
    if os.path.exists(model_local_path_or_repo_id):
        local_model_path = Path(model_local_path_or_repo_id)
        if os.path.exists(local_model_path / 'pytorch_model.bin.index.json') or \
           os.path.exists(local_model_path / 'model.safetensors.index.json') or \
           len(glob(str(local_model_path / '*.safetensors'))) > 0:
            print(f"found local checkpoint files...", file=stderr)
            return Path(model_local_path_or_repo_id), split_and_save_layers(model_local_path_or_repo_id, layer_shards_saving_path,
                                                                            compression=compression, layer_names=layer_names,
                                                                            delete_original=delete_original,
                                                                            moe_expert_sharding=moe_expert_sharding,
                                                                            moe_strict_streaming=moe_strict_streaming,
                                                                            moe_allow_dense_fallback=moe_allow_dense_fallback,
                                                                            moe_selective_fused_adapter_name=moe_selective_fused_adapter_name,
                                                                            resume_split_build=resume_split_build,
                                                                            lazy_build=lazy_build)
        else:
            print(
                f"Found local directory in {model_local_path_or_repo_id}, but didn't find downloaded model. Try using {model_local_path_or_repo_id} as a HF repo...",
                file=stderr,
            )

    # it should be a repo id at this point...
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
        #allow_patterns= ["model.safetensors.index.json", 'pytorch_model.bin.index.json'],
        ignore_patterns=['*.safetensors', '*.bin'])


    # check if there's safetensors saved, if so, exclude torch saves
    # delay download now...
    '''
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token, allow_patterns="model.safetensors.index.json")
    if len(glob(str(Path(hf_cache_path) / "model.safetensors.index.json"))) > 0:
        # there's safe tensor version, exclude torch version
        hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
                                                          ignore_patterns=['pytorch_model.bin.index.json', '*.bin'])

    else:
        hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id,
                                                          token=hf_token)
    '''

    #assert os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json') or \
    #       os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json'), \
    #       f"{hf_cache_path}/pytorch_model.bin.index.json or {hf_cache_path}/model.safetensors.index.json should exists."

    # if splitted_model subdir exists under cache use it, otherwise split and save
    return Path(hf_cache_path), split_and_save_layers(hf_cache_path, layer_shards_saving_path,
                                                      compression=compression, layer_names=layer_names,
                                                      delete_original=delete_original, repo_id=model_local_path_or_repo_id,
                                                      hf_token=hf_token, moe_expert_sharding=moe_expert_sharding,
                                                      moe_strict_streaming=moe_strict_streaming,
                                                      moe_allow_dense_fallback=moe_allow_dense_fallback,
                                                      moe_selective_fused_adapter_name=moe_selective_fused_adapter_name,
                                                      resume_split_build=resume_split_build,
                                                      lazy_build=lazy_build)
