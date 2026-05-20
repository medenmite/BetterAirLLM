"""Metadata-only probing for MoE checkpoint layouts.

The functions in this module intentionally avoid loading model tensors. They
only download small metadata files such as ``config.json`` and
``model.safetensors.index.json`` unless a caller explicitly opts into a shard
smoke test.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


GPT_OSS_DENSE_EXPERT_SUFFIXES = (
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.gate_up_proj_bias",
    ".mlp.experts.down_proj",
    ".mlp.experts.down_proj_bias",
)
GPT_OSS_PACKED_EXPERT_SUFFIXES = (
    ".mlp.experts.gate_up_proj_blocks",
    ".mlp.experts.gate_up_proj_scales",
    ".mlp.experts.gate_up_proj_bias",
    ".mlp.experts.down_proj_blocks",
    ".mlp.experts.down_proj_scales",
    ".mlp.experts.down_proj_bias",
)
GPT_OSS_EXPERT_SUFFIXES = tuple(sorted(set(GPT_OSS_DENSE_EXPERT_SUFFIXES + GPT_OSS_PACKED_EXPERT_SUFFIXES)))
GPT_OSS_ROUTER_SUFFIXES = (
    ".mlp.router.weight",
    ".mlp.router.bias",
)
QWEN35_EXPERT_SUFFIXES = (
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
)
QWEN35_ROUTER_SUFFIXES = (
    ".mlp.gate.weight",
)
QWEN35_SHARED_EXPERT_SUFFIXES = (
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
    ".mlp.shared_expert_gate.weight",
)


def _repo_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    if cache_dir:
        return cache_dir
    return os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or str(Path.cwd() / ".hf-cache")


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _download_metadata_file(model_id: str, filename: str, cache_dir: Optional[str], token: Optional[str]) -> Path:
    if Path(model_id).exists():
        local_path = Path(model_id) / filename
        if not local_path.exists():
            raise FileNotFoundError(f"{filename} was not found under {model_id}")
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for remote metadata probing. "
            "Install requirements-dev.txt first."
        ) from exc

    path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        cache_dir=_repo_cache_dir(cache_dir),
        token=token,
    )
    return Path(path)


def load_config(model_id: str, cache_dir: Optional[str] = None, token: Optional[str] = None) -> Dict[str, Any]:
    return _load_json_file(_download_metadata_file(model_id, "config.json", cache_dir, token))


def _build_safetensors_index_from_local_shards(model_id: str) -> Optional[Dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to scan local shards when model.safetensors.index.json is missing") from exc

    model_path = Path(model_id)
    shard_paths = sorted(model_path.glob("*.safetensors"))
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


def load_safetensors_index(
    model_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        return _load_json_file(_download_metadata_file(model_id, "model.safetensors.index.json", cache_dir, token))
    except FileNotFoundError:
        if Path(model_id).exists():
            return _build_safetensors_index_from_local_shards(model_id)
        return None


def _is_gpt_oss_config(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures") or []
    return (
        str(config.get("model_type", "")).lower() == "gpt_oss"
        or any("gptossforcausallm" in str(item).replace("_", "").lower() for item in architectures)
        or any("gpt-oss" in str(item).lower() for item in architectures)
    )


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("text_config") if isinstance(config, Mapping) else None
    return nested if isinstance(nested, Mapping) else config


def _is_qwen35_config(config: Mapping[str, Any]) -> bool:
    model_type = str(config.get("model_type") or "").lower()
    text_model_type = str(_text_config(config).get("model_type") or "").lower()
    return model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} or text_model_type == "qwen3_5_moe_text"


def _first_int(config: Mapping[str, Any], names: Iterable[str], default: Optional[int] = None) -> Optional[int]:
    for scope in (config, _text_config(config)):
        for name in names:
            value = scope.get(name)
            if value is not None:
                return int(value)
    return default


def _dtype_bytes(config: Mapping[str, Any]) -> float:
    quantization = config.get("quantization_config") or {}
    quant_method = str(quantization.get("quant_method", "")).lower()
    if quant_method in {"mxfp4", "fp4", "nf4"}:
        return 0.5

    dtype = str(config.get("torch_dtype") or config.get("dtype") or "").lower()
    if dtype in {"float16", "fp16", "bfloat16", "bf16"}:
        return 2.0
    if dtype in {"float32", "fp32"}:
        return 4.0
    if dtype in {"float8", "fp8"}:
        return 1.0
    return 2.0


def _candidate_names(weight_map: Mapping[str, str], suffixes: Iterable[str]) -> List[str]:
    return sorted(name for name in weight_map if any(name.endswith(suffix) for suffix in suffixes))


def _layer_prefix_from_tensor_name(name: str) -> Optional[str]:
    marker = ".mlp."
    if marker not in name:
        return None
    return name.split(marker, 1)[0]


def _expected_gpt_oss_layer_names(layer_prefix: str, packed: bool = False) -> Tuple[str, ...]:
    expert_suffixes = GPT_OSS_PACKED_EXPERT_SUFFIXES if packed else GPT_OSS_DENSE_EXPERT_SUFFIXES
    return tuple(layer_prefix + suffix for suffix in (*GPT_OSS_ROUTER_SUFFIXES, *expert_suffixes))


def _expected_qwen35_layer_names(layer_prefix: str) -> Tuple[str, ...]:
    return tuple(layer_prefix + suffix for suffix in (*QWEN35_ROUTER_SUFFIXES, *QWEN35_EXPERT_SUFFIXES))


def _qwen35_layer_complete(layer_prefix: str, weight_map: Mapping[str, str]) -> bool:
    return all(name in weight_map for name in _expected_qwen35_layer_names(layer_prefix))


def _layer_layout_kind(layer_prefix: str, weight_map: Mapping[str, str]) -> Optional[str]:
    if all(name in weight_map for name in _expected_gpt_oss_layer_names(layer_prefix, packed=False)):
        return "dense"
    if all(name in weight_map for name in _expected_gpt_oss_layer_names(layer_prefix, packed=True)):
        return "mxfp4_packed"
    return None


def _estimate_gpt_oss_bytes(config: Mapping[str, Any], index: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    hidden_size = _first_int(config, ("hidden_size", "n_embd"), 0) or 0
    intermediate_size = _first_int(config, ("intermediate_size", "ffn_dim"), 0) or 0
    num_experts = _first_int(config, ("num_local_experts", "num_experts", "n_routed_experts"), 0) or 0
    top_k = _first_int(config, ("num_experts_per_tok", "experts_per_token", "top_k"), 1) or 1
    num_layers = _first_int(config, ("num_hidden_layers", "n_layer"), 0) or 0
    dtype_bytes = _dtype_bytes(config)

    per_expert_params = (
        hidden_size * (2 * intermediate_size)
        + (2 * intermediate_size)
        + intermediate_size * hidden_size
        + hidden_size
    )
    per_expert_bytes = int(per_expert_params * dtype_bytes)
    full_expert_bytes_per_layer = int(per_expert_bytes * num_experts)
    selected_expert_bytes_per_layer_token = int(per_expert_bytes * top_k)
    avoided_expert_bytes_per_layer_token = int(per_expert_bytes * max(0, num_experts - top_k))
    selected_expert_bytes_per_token = selected_expert_bytes_per_layer_token * num_layers
    full_expert_bytes_avoided_per_token = avoided_expert_bytes_per_layer_token * num_layers
    full_fused_expert_bytes_all_layers = full_expert_bytes_per_layer * num_layers

    total_checkpoint_bytes = None
    if index:
        total_checkpoint_bytes = (index.get("metadata") or {}).get("total_size")
    estimated_dense_checkpoint_bytes = None
    if isinstance(total_checkpoint_bytes, int):
        estimated_dense_checkpoint_bytes = max(0, total_checkpoint_bytes - full_fused_expert_bytes_all_layers)

    return {
        "dtype_bytes_assumption": dtype_bytes,
        "per_expert_bytes": per_expert_bytes,
        "full_expert_bytes_per_layer": full_expert_bytes_per_layer,
        "selected_expert_bytes_per_layer_token": selected_expert_bytes_per_layer_token,
        "full_expert_bytes_avoided_per_layer_token": avoided_expert_bytes_per_layer_token,
        "selected_expert_bytes_per_token": selected_expert_bytes_per_token,
        "full_expert_bytes_avoided_per_token": full_expert_bytes_avoided_per_token,
        "full_fused_expert_bytes_all_layers": full_fused_expert_bytes_all_layers,
        "total_checkpoint_bytes_from_index": total_checkpoint_bytes,
        "estimated_dense_checkpoint_bytes": estimated_dense_checkpoint_bytes,
    }


def _estimate_qwen35_bytes(config: Mapping[str, Any], index: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    text_config = _text_config(config)
    hidden_size = _first_int(text_config, ("hidden_size", "n_embd"), 0) or 0
    intermediate_size = _first_int(text_config, ("moe_intermediate_size",), 0) or 0
    num_experts = _first_int(text_config, ("num_experts", "num_local_experts", "n_routed_experts"), 0) or 0
    top_k = _first_int(text_config, ("num_experts_per_tok", "experts_per_token", "top_k"), 1) or 1
    num_layers = _first_int(text_config, ("num_hidden_layers", "n_layer"), 0) or 0
    dtype_bytes = _dtype_bytes(config)

    per_expert_params = hidden_size * (2 * intermediate_size) + hidden_size * intermediate_size
    per_expert_bytes = int(per_expert_params * dtype_bytes)
    full_expert_bytes_per_layer = int(per_expert_bytes * num_experts)
    selected_expert_bytes_per_layer_token = int(per_expert_bytes * top_k)
    avoided_expert_bytes_per_layer_token = int(per_expert_bytes * max(0, num_experts - top_k))
    full_fused_expert_bytes_all_layers = full_expert_bytes_per_layer * num_layers

    total_checkpoint_bytes = None
    if index:
        total_checkpoint_bytes = (index.get("metadata") or {}).get("total_size")
    estimated_dense_checkpoint_bytes = None
    if isinstance(total_checkpoint_bytes, int):
        estimated_dense_checkpoint_bytes = max(0, total_checkpoint_bytes - full_fused_expert_bytes_all_layers)

    return {
        "dtype_bytes_assumption": dtype_bytes,
        "per_expert_bytes": per_expert_bytes,
        "full_expert_bytes_per_layer": full_expert_bytes_per_layer,
        "selected_expert_bytes_per_layer_token": selected_expert_bytes_per_layer_token,
        "full_expert_bytes_avoided_per_layer_token": avoided_expert_bytes_per_layer_token,
        "selected_expert_bytes_per_token": selected_expert_bytes_per_layer_token * num_layers,
        "full_expert_bytes_avoided_per_token": avoided_expert_bytes_per_layer_token * num_layers,
        "full_fused_expert_bytes_all_layers": full_fused_expert_bytes_all_layers,
        "total_checkpoint_bytes_from_index": total_checkpoint_bytes,
        "estimated_dense_checkpoint_bytes": estimated_dense_checkpoint_bytes,
    }


def probe_gpt_oss_layout(
    model_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    config = load_config(model_id, cache_dir=cache_dir, token=token)
    index = load_safetensors_index(model_id, cache_dir=cache_dir, token=token)
    weight_map = (index or {}).get("weight_map") or {}

    router_names = _candidate_names(weight_map, GPT_OSS_ROUTER_SUFFIXES)
    expert_names = _candidate_names(weight_map, GPT_OSS_EXPERT_SUFFIXES)
    layer_prefixes = sorted({prefix for name in expert_names if (prefix := _layer_prefix_from_tensor_name(name))})

    expected_by_layer = {}
    for prefix in layer_prefixes[:3]:
        dense_expected = _expected_gpt_oss_layer_names(prefix, packed=False)
        packed_expected = _expected_gpt_oss_layer_names(prefix, packed=True)
        expected_by_layer[prefix] = {
            "layout_kind": _layer_layout_kind(prefix, weight_map),
            "dense_expected": list(dense_expected),
            "dense_missing": [name for name in dense_expected if name not in weight_map],
            "packed_expected": list(packed_expected),
            "packed_missing": [name for name in packed_expected if name not in weight_map],
        }
    complete_layers = [prefix for prefix in layer_prefixes if _layer_layout_kind(prefix, weight_map)]
    dense_complete_layers = [prefix for prefix in layer_prefixes if _layer_layout_kind(prefix, weight_map) == "dense"]
    packed_complete_layers = [prefix for prefix in layer_prefixes if _layer_layout_kind(prefix, weight_map) == "mxfp4_packed"]
    model_type = str(config.get("model_type") or "").lower()
    architectures = config.get("architectures") or []
    looks_like_gpt_oss = model_type == "gpt_oss" or any("gptoss" in str(arch).lower().replace("_", "") for arch in architectures)
    metadata_matches = bool(looks_like_gpt_oss and complete_layers and router_names and expert_names)
    dense_adapter_matches = bool(looks_like_gpt_oss and dense_complete_layers and router_names)
    packed_adapter_metadata_matches = bool(looks_like_gpt_oss and packed_complete_layers and router_names)

    unsupported_reasons = []
    if not looks_like_gpt_oss:
        unsupported_reasons.append(f"model_type/architectures do not identify GPT-OSS: {model_type or architectures}")
    if not index:
        unsupported_reasons.append("model.safetensors.index.json is missing")
    if not router_names:
        unsupported_reasons.append("no GPT-OSS router tensor names were found")
    if not expert_names:
        unsupported_reasons.append("no GPT-OSS fused expert tensor names were found")
    if layer_prefixes and not complete_layers:
        unsupported_reasons.append("no layer has the full expected router/expert tensor set")

    adapter_name = "gpt_oss_mxfp4_reference" if packed_adapter_metadata_matches else "gpt_oss"
    manifest = build_manifest_verification(config, weight_map, adapter_name=adapter_name)
    return {
        "model_id": model_id,
        "architecture": architectures,
        "model_type": config.get("model_type"),
        "num_hidden_layers": _first_int(config, ("num_hidden_layers", "n_layer")),
        "num_experts": _first_int(config, ("num_local_experts", "num_experts", "n_routed_experts")),
        "experts_per_token": _first_int(config, ("num_experts_per_tok", "experts_per_token", "top_k")),
        "hidden_size": _first_int(config, ("hidden_size", "n_embd")),
        "intermediate_size": _first_int(config, ("intermediate_size", "ffn_dim")),
        "candidate_router_tensor_names": router_names[:16],
        "candidate_router_tensor_count": len(router_names),
        "candidate_expert_tensor_names": expert_names[:16],
        "candidate_expert_tensor_count": len(expert_names),
        "sample_expected_layer_tensors": expected_by_layer,
        "adapter_expectations_match": metadata_matches,
        "dense_adapter_expectations_match": dense_adapter_matches,
        "packed_adapter_metadata_match": packed_adapter_metadata_matches,
        "gpt_oss_weight_encoding": "mxfp4_packed" if packed_adapter_metadata_matches else ("dense" if dense_adapter_matches else "unknown"),
        "strict_selective_runtime_can_enable": bool(metadata_matches and manifest["manifest_checks_pass"]),
        "strict_selective_runtime_blockers": unsupported_reasons + manifest["failures"],
        "manifest_verification": manifest,
        "estimates": _estimate_gpt_oss_bytes(config, index),
    }


def probe_qwen35_layout(
    model_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    config = load_config(model_id, cache_dir=cache_dir, token=token)
    index = load_safetensors_index(model_id, cache_dir=cache_dir, token=token)
    weight_map = (index or {}).get("weight_map") or {}
    text_config = _text_config(config)

    router_names = _candidate_names(weight_map, QWEN35_ROUTER_SUFFIXES)
    expert_names = _candidate_names(weight_map, QWEN35_EXPERT_SUFFIXES)
    shared_names = _candidate_names(weight_map, QWEN35_SHARED_EXPERT_SUFFIXES)
    layer_prefixes = sorted({prefix for name in expert_names if (prefix := _layer_prefix_from_tensor_name(name))})

    expected_by_layer = {}
    for prefix in layer_prefixes[:3]:
        expected = _expected_qwen35_layer_names(prefix)
        expected_by_layer[prefix] = {
            "layout_kind": "qwen3_5_moe_fused" if _qwen35_layer_complete(prefix, weight_map) else "incomplete",
            "expected": list(expected),
            "missing": [name for name in expected if name not in weight_map],
            "shared_expert_present": any(name.startswith(prefix + ".mlp.shared_expert") for name in shared_names),
        }

    complete_layers = [prefix for prefix in layer_prefixes if _qwen35_layer_complete(prefix, weight_map)]
    looks_like_qwen35 = _is_qwen35_config(config)
    metadata_matches = bool(looks_like_qwen35 and complete_layers and router_names and expert_names)

    unsupported_reasons = []
    if not looks_like_qwen35:
        unsupported_reasons.append(
            f"model_type/text_config.model_type do not identify qwen3_5_moe: "
            f"{config.get('model_type')!r}/{text_config.get('model_type')!r}"
        )
    if not index:
        unsupported_reasons.append("model.safetensors.index.json is missing")
    if not router_names:
        unsupported_reasons.append("no Qwen3.5/Qwen3.6 router tensor names were found")
    if not expert_names:
        unsupported_reasons.append("no Qwen3.5/Qwen3.6 fused routed expert tensor names were found")
    if layer_prefixes and not complete_layers:
        unsupported_reasons.append("no layer has the full expected Qwen routed expert tensor set")
    if not shared_names:
        unsupported_reasons.append("shared expert tensors were not found; runtime expects them to remain dense")

    manifest = build_manifest_verification(config, weight_map, adapter_name="qwen3_5_moe")
    return {
        "model_id": model_id,
        "architecture": config.get("architectures") or [],
        "model_type": config.get("model_type"),
        "text_model_type": text_config.get("model_type"),
        "num_hidden_layers": _first_int(text_config, ("num_hidden_layers", "n_layer")),
        "num_experts": _first_int(text_config, ("num_experts", "num_local_experts", "n_routed_experts")),
        "experts_per_token": _first_int(text_config, ("num_experts_per_tok", "experts_per_token", "top_k")),
        "hidden_size": _first_int(text_config, ("hidden_size", "n_embd")),
        "moe_intermediate_size": _first_int(text_config, ("moe_intermediate_size",)),
        "candidate_router_tensor_names": router_names[:16],
        "candidate_router_tensor_count": len(router_names),
        "candidate_expert_tensor_names": expert_names[:16],
        "candidate_expert_tensor_count": len(expert_names),
        "candidate_shared_expert_tensor_names": shared_names[:16],
        "candidate_shared_expert_tensor_count": len(shared_names),
        "sample_expected_layer_tensors": expected_by_layer,
        "has_shared_expert": bool(shared_names),
        "adapter_expectations_match": metadata_matches,
        "qwen3_5_moe_weight_encoding": "fused_routed_experts",
        "strict_selective_runtime_can_enable": bool(metadata_matches and manifest["manifest_checks_pass"]),
        "strict_selective_runtime_blockers": unsupported_reasons + manifest["failures"],
        "manifest_verification": manifest,
        "estimates": _estimate_qwen35_bytes(config, index),
    }


def probe_moe_layout(
    model_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    config = load_config(model_id, cache_dir=cache_dir, token=token)
    if _is_qwen35_config(config):
        return probe_qwen35_layout(model_id, cache_dir=cache_dir, token=token)
    return probe_gpt_oss_layout(model_id, cache_dir=cache_dir, token=token)


def build_manifest_verification(
    config: Mapping[str, Any],
    weight_map: Mapping[str, str],
    adapter_name: Optional[str],
) -> Dict[str, Any]:
    if adapter_name == "qwen3_5_moe" or _is_qwen35_config(config):
        router_names = _candidate_names(weight_map, QWEN35_ROUTER_SUFFIXES)
        expert_names = _candidate_names(weight_map, QWEN35_EXPERT_SUFFIXES)
        shared_names = _candidate_names(weight_map, QWEN35_SHARED_EXPERT_SUFFIXES)
        layer_prefixes = sorted({prefix for name in expert_names if (prefix := _layer_prefix_from_tensor_name(name))})
        complete_layers = [prefix for prefix in layer_prefixes if _qwen35_layer_complete(prefix, weight_map)]
        has_fused_experts = bool(expert_names)
        fused_experts_split = bool(complete_layers)
        valid_adapter = adapter_name == "qwen3_5_moe"
        shared_expert_remains_dense = bool(shared_names)
        selective_fused_runtime = bool(valid_adapter and has_fused_experts and fused_experts_split and router_names)
        dense_contains_full_fused_experts = not selective_fused_runtime
        requires_dense_fallback = bool(has_fused_experts and not selective_fused_runtime)

        failures = []
        if not has_fused_experts:
            failures.append("Qwen routed fused expert tensor names were not detected")
        if not fused_experts_split:
            failures.append("Qwen routed fused experts cannot be split by metadata expectations")
        if not valid_adapter:
            failures.append("strict Qwen selective runtime requires the qwen3_5_moe adapter")
        if not router_names:
            failures.append("Qwen router tensor names were not detected")
        if not shared_expert_remains_dense:
            failures.append("Qwen shared expert tensors were not detected in dense layer metadata")
        if dense_contains_full_fused_experts:
            failures.append("strict adapter mode would still keep full routed fused expert tensors in dense shards")
        if requires_dense_fallback:
            failures.append("strict adapter mode would require dense fallback")

        return {
            "moe_layout": "qwen3_5_moe" if _is_qwen35_config(config) else "unknown",
            "has_fused_experts": has_fused_experts,
            "fused_experts_split": fused_experts_split,
            "selective_fused_runtime": selective_fused_runtime,
            "dense_contains_full_fused_experts": dense_contains_full_fused_experts,
            "requires_dense_fallback": requires_dense_fallback,
            "adapter_name": adapter_name,
            "mxfp4_execution": None,
            "shared_expert_remains_dense": shared_expert_remains_dense,
            "complete_moe_layer_count": len(complete_layers),
            "fused_tensor_name_count": len(expert_names),
            "router_tensor_name_count": len(router_names),
            "shared_expert_tensor_name_count": len(shared_names),
            "manifest_checks_pass": bool(
                selective_fused_runtime
                and not dense_contains_full_fused_experts
                and not requires_dense_fallback
                and shared_expert_remains_dense
            ),
            "failures": failures,
        }

    router_names = _candidate_names(weight_map, GPT_OSS_ROUTER_SUFFIXES)
    expert_names = _candidate_names(weight_map, GPT_OSS_EXPERT_SUFFIXES)
    layer_prefixes = sorted({prefix for name in expert_names if (prefix := _layer_prefix_from_tensor_name(name))})
    by_layer = defaultdict(set)
    for name in expert_names:
        prefix = _layer_prefix_from_tensor_name(name)
        if prefix:
            by_layer[prefix].add(name)

    has_fused_experts = bool(expert_names)
    complete_layers = [prefix for prefix in layer_prefixes if _layer_layout_kind(prefix, weight_map)]
    fused_experts_split = bool(complete_layers)
    valid_adapter = adapter_name in {"gpt_oss", "gpt_oss_mxfp4_reference"}
    selective_fused_runtime = bool(valid_adapter and has_fused_experts and fused_experts_split and router_names)
    dense_contains_full_fused_experts = not selective_fused_runtime
    requires_dense_fallback = bool(has_fused_experts and not selective_fused_runtime)

    failures = []
    if not has_fused_experts:
        failures.append("fused expert tensor names were not detected")
    if not fused_experts_split:
        failures.append("fused expert tensors cannot be split by GPT-OSS metadata expectations")
    if not valid_adapter:
        failures.append("strict GPT-OSS selective runtime requires a registered GPT-OSS adapter")
    if not router_names:
        failures.append("router tensor names were not detected")
    if dense_contains_full_fused_experts:
        failures.append("strict adapter mode would still keep full fused expert tensors in dense shards")
    if requires_dense_fallback:
        failures.append("strict adapter mode would require dense fallback")

    return {
        "moe_layout": "gpt_oss" if str(config.get("model_type", "")).lower() == "gpt_oss" else "unknown",
        "has_fused_experts": has_fused_experts,
        "fused_experts_split": fused_experts_split,
        "selective_fused_runtime": selective_fused_runtime,
        "dense_contains_full_fused_experts": dense_contains_full_fused_experts,
        "requires_dense_fallback": requires_dense_fallback,
        "adapter_name": adapter_name,
        "mxfp4_execution": "reference_dequant" if adapter_name == "gpt_oss_mxfp4_reference" else None,
        "complete_moe_layer_count": len(complete_layers),
        "fused_tensor_name_count": len(expert_names),
        "router_tensor_name_count": len(router_names),
        "manifest_checks_pass": bool(selective_fused_runtime and not dense_contains_full_fused_experts and not requires_dense_fallback),
        "failures": failures,
    }


def dry_run_gb_per_token_report(
    model_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    vram_cache_mb: Optional[int] = None,
    cpu_cache_mb: int = 0,
    os_reserved_ram_mb: int = 8192,
) -> Dict[str, Any]:
    probe = probe_gpt_oss_layout(model_id, cache_dir=cache_dir, token=token)
    estimates = probe["estimates"]
    selected = int(estimates["selected_expert_bytes_per_token"] or 0)
    avoided = int(estimates["full_expert_bytes_avoided_per_token"] or 0)
    dense = selected + avoided
    cache_bytes = int((vram_cache_mb or 0) * 1024 ** 2 + (cpu_cache_mb or 0) * 1024 ** 2)

    return {
        "model": model_id,
        "runtime": "dry_run_layout",
        "strict_selective_runtime_can_enable": probe["strict_selective_runtime_can_enable"],
        "blockers": probe["strict_selective_runtime_blockers"],
        "top_k": probe["experts_per_token"],
        "dense_gb_per_token": dense / 1024 ** 3,
        "selective_gb_per_token": selected / 1024 ** 3,
        "selected_expert_bytes_per_token": selected,
        "full_expert_bytes_avoided": avoided,
        "bytes_avoided": avoided,
        "expected_cache_pressure": {
            "cache_budget_mb": (vram_cache_mb or 0) + cpu_cache_mb,
            "selected_expert_bytes_per_token": selected,
            "selected_expert_bytes_fit_in_cache_budget": selected <= cache_bytes if cache_bytes else False,
        },
        "expected_ram_pressure": {
            "os_reserved_ram_mb": os_reserved_ram_mb,
            "estimated_dense_checkpoint_gb": (
                estimates["estimated_dense_checkpoint_bytes"] / 1024 ** 3
                if estimates["estimated_dense_checkpoint_bytes"] is not None else None
            ),
            "selected_expert_gb_per_token": selected / 1024 ** 3,
        },
        "expected_vram_pressure": {
            "vram_cache_mb": vram_cache_mb,
            "selected_expert_gb_per_token": selected / 1024 ** 3,
        },
        "probe": probe,
    }


def download_gpt_oss_one_expert_mxfp4_smoke(
    model_id: str,
    *,
    layer_id: int = 0,
    expert_id: int = 0,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    dtype_name: str = "bfloat16",
) -> Dict[str, Any]:
    """Download the shard(s) needed for one GPT-OSS expert and dequantize it.

    Hugging Face stores tensors by shard file, so "minimum" means the minimum
    checkpoint shard files containing this layer's packed expert tensors, while
    ``safe_open.get_slice`` is used to materialize only one expert row.
    """
    config = load_config(model_id, cache_dir=cache_dir, token=token)
    if not _is_gpt_oss_config(config):
        raise ValueError(
            "one-expert MXFP4 smoke is only enabled for GPT-OSS-style models. "
            f"config model_type={config.get('model_type')!r}, architectures={config.get('architectures')!r}"
        )

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "one-expert MXFP4 smoke requires torch and safetensors"
        ) from exc

    from .gpt_oss_mxfp4 import dequantize_mxfp4_expert, mxfp4_state_nbytes, tensor_nbytes

    index = load_safetensors_index(model_id, cache_dir=cache_dir, token=token)
    if not index:
        raise FileNotFoundError("model.safetensors.index.json is required for one-expert smoke")
    weight_map = index.get("weight_map") or {}
    layer_prefix = f"model.layers.{int(layer_id)}"
    tensor_names = [layer_prefix + suffix for suffix in GPT_OSS_PACKED_EXPERT_SUFFIXES]
    missing = [name for name in tensor_names if name not in weight_map]
    if missing:
        raise KeyError(f"missing GPT-OSS packed tensors for {layer_prefix}: {missing}")

    shard_names = sorted({weight_map[name] for name in tensor_names})
    local_root = Path(model_id).expanduser() if Path(model_id).expanduser().is_dir() else None
    if local_root is not None:
        shard_paths = {shard: local_root / shard for shard in shard_names}
        missing_shards = [shard for shard, path in shard_paths.items() if not path.exists()]
        if missing_shards:
            raise FileNotFoundError(f"missing local safetensors shard(s): {missing_shards}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for remote one-expert MXFP4 smoke"
            ) from exc
        shard_paths = {
            shard: Path(hf_hub_download(model_id, shard, cache_dir=_repo_cache_dir(cache_dir), token=token))
            for shard in shard_names
        }

    expert_state = {}
    tensor_shapes = {}
    for name in tensor_names:
        mapped_name = name.rsplit(".", 1)[-1]
        with safe_open(str(shard_paths[weight_map[name]]), framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(name)
            tensor_shapes[name] = list(tensor_slice.get_shape())
            expert_state[mapped_name] = tensor_slice[int(expert_id)].contiguous()

    dtype = torch.bfloat16 if dtype_name in {"bfloat16", "bf16"} else torch.float16
    dense = dequantize_mxfp4_expert(expert_state, dtype=dtype, device="cpu")
    packed_bytes_loaded = mxfp4_state_nbytes(expert_state)
    dense_temporary_bytes = sum(tensor_nbytes(tensor) for tensor in dense.values())
    full_layer_packed_bytes_estimate = packed_bytes_loaded * int(config.get("num_local_experts", 1))

    return {
        "one_expert_smoke": "mxfp4_dequantized",
        "model": model_id,
        "layer_id": int(layer_id),
        "expert_id": int(expert_id),
        "downloaded_shards": shard_names,
        "tensor_shapes": tensor_shapes,
        "selected_expert_tensor_shapes": {name: list(tensor.shape) for name, tensor in expert_state.items()},
        "dequantized_tensor_shapes": {name: list(tensor.shape) for name, tensor in dense.items()},
        "dequantized_dtype": str(dtype),
        "packed_bytes_loaded": packed_bytes_loaded,
        "dequantized_temporary_bytes": dense_temporary_bytes,
        "estimated_full_layer_packed_bytes": full_layer_packed_bytes_estimate,
        "estimated_dense_bytes_avoided": max(0, full_layer_packed_bytes_estimate - packed_bytes_loaded),
        "strict_full_fused_tensor_loaded": False,
    }
