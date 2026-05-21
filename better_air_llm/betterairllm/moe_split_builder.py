from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .utils import infer_num_experts_from_config, load_checkpoint_weight_map, split_moe_layer_state_dict


SPLIT_FORMAT_VERSION = 1
_ATOMIC_WRITE_LOCK = threading.RLock()


class MissingCheckpointShardError(FileNotFoundError):
    pass


def _now() -> float:
    return time.time()


def _json_read(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_write_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    with _ATOMIC_WRITE_LOCK:
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()


def _atomic_tmp_path(path: Path) -> Path:
    return path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )


def _state_nbytes(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class IncrementalMoESplitBuilder:

    def __init__(
        self,
        checkpoint_path,
        saving_path,
        *,
        layer_names=None,
        repo_id=None,
        hf_token=None,
        strict_mode=True,
        adapter_name=None,
        allow_dense_fallback=False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.saving_path = Path(saving_path)
        self.layer_names = layer_names
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.strict_mode = bool(strict_mode)
        self.adapter_name = adapter_name
        self.allow_dense_fallback = bool(allow_dense_fallback)
        self.num_experts = infer_num_experts_from_config(self.checkpoint_path)
        self.saving_path.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.saving_path / "progress.json"
        self.manifest_path = self.saving_path / "manifest.json"
        self.moe_manifest_path = self.saving_path / "moe_expert_index.json"
        self.index = self._load_index()
        self.layers = self._build_layers()
        self._progress_lock = threading.RLock()
        self.progress = self._load_or_init_progress()
        self._ensure_progress_schema()
        self._write_manifest()
        self._write_progress()
        self._write_moe_manifest()

    def _load_index(self) -> Dict[str, str]:
        weight_map, _ = load_checkpoint_weight_map(self.checkpoint_path)
        return weight_map

    def _build_layers(self):
        if self.layer_names is None:
            n_layers = len({int(k.split(".")[2]) for k in self.index if "model.layers" in k})
            return ["model.embed_tokens."] + [f"model.layers.{i}." for i in range(n_layers)] + ["model.norm.", "lm_head."]

        n_layers = len({
            int(k[len(self.layer_names["layer_prefix"]):].split(".")[1])
            for k in self.index
            if self.layer_names["layer_prefix"] in k
        })
        layers = [self.layer_names["embed"]] + [
            f'{self.layer_names["layer_prefix"]}.{i}' for i in range(n_layers)
        ] + [self.layer_names["norm"], self.layer_names["lm_head"]]
        if "rotary_pos_emb" in self.layer_names:
            layers = [self.layer_names["rotary_pos_emb"]] + layers
        return [layer + "." for layer in layers]

    def _source_hash(self):
        digest = hashlib.sha256()
        for name in sorted(set(self.index.values())):
            digest.update(name.encode("utf-8"))
            path = self.checkpoint_path / name
            if path.exists():
                digest.update(str(path.stat().st_size).encode("ascii"))
        return digest.hexdigest()

    def _load_or_init_progress(self):
        progress = _json_read(self.progress_path, None)
        if progress:
            progress["interrupted"] = True
            progress["updated_at"] = _now()
            return progress
        total_experts = (len([layer for layer in self.layers if ".layers." in layer]) * int(self.num_experts or 0))
        return {
            "total_layers": len(self.layers),
            "completed_layers": [],
            "dense_layers_ready": [],
            "router_layers_ready": [],
            "total_experts": total_experts,
            "completed_experts": [],
            "experts_ready_total": [],
            "experts_ready_selected_for_current_run": [],
            "experts_ready_background": [],
            "experts_unused_built": [],
            "selected_expert_builds_this_run": 0,
            "unused_expert_builds_this_run": 0,
            "direct_slice_expert_loads_this_run": 0,
            "direct_slice_packed_bytes_read_this_run": 0,
            "selected_expert_materialization_time_this_run": 0.0,
            "completed_dense_shards": [],
            "bytes_processed": 0,
            "estimated_remaining_bytes": self._estimate_remaining_bytes(set(), set()),
            "started_at": _now(),
            "updated_at": _now(),
            "interrupted": False,
            "strict_mode": self.strict_mode,
            "adapter_name": self.adapter_name,
            "split_format_version": SPLIT_FORMAT_VERSION,
            "layer_timings": {},
        }

    def _ensure_progress_schema(self):
        defaults = {
            "completed_layers": [],
            "dense_layers_ready": [],
            "router_layers_ready": [],
            "completed_experts": [],
            "experts_ready_total": [],
            "experts_ready_selected_for_current_run": [],
            "experts_ready_background": [],
            "experts_unused_built": [],
            "selected_expert_builds_this_run": 0,
            "unused_expert_builds_this_run": 0,
            "direct_slice_expert_loads_this_run": 0,
            "direct_slice_packed_bytes_read_this_run": 0,
            "selected_expert_materialization_time_this_run": 0.0,
            "completed_dense_shards": [],
            "layer_timings": {},
        }
        for key, value in defaults.items():
            self.progress.setdefault(key, value.copy() if isinstance(value, list) else value)
        self.progress["experts_ready_selected_for_current_run"] = []
        self.progress["selected_expert_builds_this_run"] = 0
        self.progress["unused_expert_builds_this_run"] = 0
        self.progress["direct_slice_expert_loads_this_run"] = 0
        self.progress["direct_slice_packed_bytes_read_this_run"] = 0
        self.progress["selected_expert_materialization_time_this_run"] = 0.0

    def reset_run_metrics(self):
        self.progress["experts_ready_selected_for_current_run"] = []
        self.progress["selected_expert_builds_this_run"] = 0
        self.progress["unused_expert_builds_this_run"] = 0
        self.progress["direct_slice_expert_loads_this_run"] = 0
        self.progress["direct_slice_packed_bytes_read_this_run"] = 0
        self.progress["selected_expert_materialization_time_this_run"] = 0.0
        self.progress["layer_timings"] = {}
        self.progress["updated_at"] = _now()
        self._write_progress()

    def _write_manifest(self):
        manifest = {
            "source_model_id": self.repo_id,
            "source_checkpoint_path": str(self.checkpoint_path),
            "source_checkpoint_hash": self._source_hash(),
            "split_format_version": SPLIT_FORMAT_VERSION,
            "adapter_name": self.adapter_name,
            "strict_mode": self.strict_mode,
            "allow_dense_fallback": self.allow_dense_fallback,
            "lazy_ready": True,
        }
        with self._progress_lock:
            _json_write_atomic(self.manifest_path, manifest)

    def _write_progress(self):
        with self._progress_lock:
            completed_layers = set(self.progress.get("completed_layers", []))
            completed_experts = set(self.progress.get("completed_experts", []))
            self.progress["estimated_remaining_bytes"] = self._estimate_remaining_bytes(completed_layers, completed_experts)
            self.progress["interrupted"] = False
            self.progress["updated_at"] = _now()
            _json_write_atomic(self.progress_path, dict(self.progress))

    def _estimate_remaining_bytes(self, completed_layers: Set[str], completed_experts: Set[str]) -> int:
        remaining_shards = set()
        for layer in self.layers:
            if layer not in completed_layers:
                remaining_shards.update(self._layer_shards(layer))
        return sum(
            (self.checkpoint_path / shard).stat().st_size
            for shard in remaining_shards
            if (self.checkpoint_path / shard).exists()
        )

    def _layer_shards(self, layer: str):
        return sorted({v for k, v in self.index.items() if k.startswith(layer)})

    def _layer_keys(self, layer: str):
        return sorted(k for k in self.index if k.startswith(layer))

    def _ensure_source_shard(self, shard: str, *, tensor_key: Optional[str] = None) -> Path:
        shard_path = self.checkpoint_path / shard
        if shard_path.exists():
            return shard_path
        if self.repo_id is None:
            raise MissingCheckpointShardError(self._missing_shard_message(shard, tensor_key))
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(self.repo_id, shard, token=self.hf_token)
            shard_path = Path(downloaded)
        except Exception as exc:
            raise MissingCheckpointShardError(self._missing_shard_message(shard, tensor_key)) from exc
        if not shard_path.exists():
            raise MissingCheckpointShardError(self._missing_shard_message(shard, tensor_key))
        return shard_path

    def _missing_shard_message(self, shard: str, tensor_key: Optional[str] = None) -> str:
        why = f"tensor key {tensor_key}" if tensor_key else "a requested tensor"
        model = self.repo_id or self.checkpoint_path
        return (
            f"Missing original checkpoint shard: {shard}. It is needed for {why}. "
            "Prefetch it before decode with: "
            f"python scripts/smoke_gpt_oss_20b_one_token.py {model} --prefetch-required-shards --confirm-download. "
            "The split/decode state is resumable after the shard download completes."
        )

    def _flat_stem(self, layer_or_expert: str) -> str:
        return layer_or_expert.rstrip(".")

    def _shard_path(self, layer_or_expert: str) -> Path:
        return self.saving_path / (self._flat_stem(layer_or_expert) + ".safetensors")

    def _done_path(self, layer_or_expert: str) -> Path:
        return self.saving_path / (self._flat_stem(layer_or_expert) + ".safetensors.done")

    @staticmethod
    def _normalize_layer(layer_name: str) -> str:
        return layer_name if layer_name.endswith(".") else layer_name + "."

    @staticmethod
    def _parse_expert_shard_name(shard_name: str):
        normalized = shard_name.rstrip(".")
        if ".experts." not in normalized:
            raise ValueError(f"Expected expert shard name, got {shard_name!r}")
        layer_part, expert_part = normalized.rsplit(".mlp.experts.", 1)
        expert_id = int(expert_part.split(".", 1)[0])
        return layer_part + ".", expert_id, normalized

    def _expected_expert_prefixes(self, layer: str):
        if ".layers." not in layer or not self.num_experts:
            return []
        base = layer.rstrip(".")
        return [f"{base}.mlp.experts.{idx}" for idx in range(int(self.num_experts))]

    def _valid_saved(self, name: str) -> bool:
        path = self._shard_path(name)
        done = self._done_path(name)
        if not path.exists() or not done.exists() or path.stat().st_size <= 0:
            return False
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                _ = list(handle.keys())
            return True
        except Exception:
            return False

    def _save_state(self, state_dict: Dict[str, torch.Tensor], name: str) -> int:
        path = self._shard_path(name)
        done = self._done_path(name)
        tmp = _atomic_tmp_path(path)
        try:
            save_file(state_dict, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        done.touch()
        return path.stat().st_size

    def ensure_layer(self, layer_name: str) -> Dict:
        layer = self._normalize_layer(layer_name)
        if ".experts." in layer:
            prefix = layer.rsplit(".experts.", 1)[0] + "."
            self.ensure_layer(prefix)
            if not self._valid_saved(layer):
                raise FileNotFoundError(f"Expected expert shard was not created: {layer}")
            return self.status()

        if self._valid_saved(layer):
            expected_experts = self._expected_expert_prefixes(layer)
            missing_experts = [
                expert_prefix
                for expert_prefix in expected_experts
                if not self._valid_saved(expert_prefix + ".")
            ]
            if missing_experts:
                self.build_layer(layer)
                return self.status()
            if layer not in self.progress["completed_layers"]:
                self.progress["completed_layers"].append(layer)
                self.progress["completed_dense_shards"].append(layer)
            for expert_prefix in expected_experts:
                if expert_prefix not in self.progress["completed_experts"]:
                    self.progress["completed_experts"].append(expert_prefix)
            if expected_experts:
                self._write_moe_manifest()
            self._write_progress()
            return self.status()

        self.build_layer(layer)
        return self.status()

    def ensure_layer_dense_ready(self, layer_name: str) -> Dict:
        layer = self._normalize_layer(layer_name)
        if ".experts." in layer:
            raise ValueError(f"Expected dense layer name, got {layer_name!r}")
        if self._valid_saved(layer):
            if layer not in self.progress["dense_layers_ready"]:
                self.progress["dense_layers_ready"].append(layer)
            if layer not in self.progress["completed_dense_shards"]:
                self.progress["completed_dense_shards"].append(layer)
            if any(
                key.startswith(layer + "mlp.router.")
                or key.startswith(layer + "mlp.gate.")
                or key.startswith(layer + "block_sparse_moe.gate.")
                for key in self._layer_keys(layer)
            ):
                if layer not in self.progress["router_layers_ready"]:
                    self.progress["router_layers_ready"].append(layer)
            self._write_progress()
            return self.status()

        started = time.perf_counter()
        dense_state_dict = {}
        bytes_read = 0
        for key in self._layer_keys(layer):
            if ".experts." in key:
                continue
            shard_path = self._ensure_source_shard(self.index[key], tensor_key=key)
            if shard_path.suffix != ".safetensors":
                raise NotImplementedError("Incremental dense split building currently requires safetensors checkpoints.")
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(key)
            dense_state_dict[key] = tensor
            bytes_read += tensor.numel() * tensor.element_size()
        if any(".experts." in key for key in dense_state_dict):
            raise RuntimeError(f"Dense shard for {layer} contains fused expert tensors.")

        dense_bytes = self._save_state(dense_state_dict, layer)
        if layer not in self.progress["dense_layers_ready"]:
            self.progress["dense_layers_ready"].append(layer)
        if layer not in self.progress["completed_dense_shards"]:
            self.progress["completed_dense_shards"].append(layer)
        if any(key.startswith(layer + "mlp.router.") or key.startswith(layer + "mlp.gate.") for key in dense_state_dict):
            if layer not in self.progress["router_layers_ready"]:
                self.progress["router_layers_ready"].append(layer)
        self.progress["bytes_processed"] += bytes_read
        elapsed = time.perf_counter() - started
        self.progress["layer_timings"].setdefault(layer, {})
        self.progress["layer_timings"][layer]["dense_ready"] = {
            "seconds": elapsed,
            "bytes_read": bytes_read,
            "dense_bytes_written": dense_bytes,
            "bytes_per_second": bytes_read / elapsed if elapsed > 0 else 0,
        }
        self._write_progress()
        self._write_moe_manifest()
        return self.status()

    def ensure_expert_ready(self, layer_id, expert_id, *, selected_for_current_run=True) -> Dict:
        if isinstance(layer_id, str):
            layer = self._normalize_layer(layer_id)
        else:
            layer = f"model.layers.{int(layer_id)}."
        expert_prefix = f"{layer.rstrip('.')}.mlp.experts.{int(expert_id)}"
        if self._valid_saved(expert_prefix + "."):
            self._mark_expert_ready(expert_prefix, selected_for_current_run=selected_for_current_run, newly_built=False)
            self._write_progress()
            self._write_moe_manifest()
            return self.status()

        started = time.perf_counter()
        expert_state = self._slice_selected_expert_state(layer, int(expert_id), prefixed=True)
        bytes_written = self._save_state(expert_state, expert_prefix + ".")
        elapsed = time.perf_counter() - started
        self._mark_expert_ready(expert_prefix, selected_for_current_run=selected_for_current_run, newly_built=True)
        self.progress["bytes_processed"] += sum(t.numel() * t.element_size() for t in expert_state.values())
        self.progress["layer_timings"].setdefault(layer, {})
        self.progress["layer_timings"][layer].setdefault("selected_experts", {})[str(expert_id)] = {
            "seconds": elapsed,
            "bytes_written": bytes_written,
            "mode": "persisted_split",
        }
        self._write_progress()
        self._write_moe_manifest()
        return self.status()

    def ensure_selected_experts_ready(self, layer_id, expert_ids) -> Dict:
        for expert_id in sorted({int(expert_id) for expert_id in expert_ids}):
            self.ensure_expert_ready(layer_id, expert_id, selected_for_current_run=True)
        return self.status()

    def load_gpt_oss_expert_direct(self, shard_name: str) -> Dict[str, torch.Tensor]:
        layer, expert_id, expert_prefix = self._parse_expert_shard_name(shard_name)
        started = time.perf_counter()
        expert_state = self._slice_gpt_oss_expert_state(layer, expert_id, prefixed=False)
        elapsed = time.perf_counter() - started
        packed_bytes = sum(t.numel() * t.element_size() for t in expert_state.values())
        self._mark_expert_ready(expert_prefix, selected_for_current_run=True, newly_built=True, direct_slice=True)
        self.progress["direct_slice_expert_loads_this_run"] += 1
        self.progress["direct_slice_packed_bytes_read_this_run"] += packed_bytes
        self.progress["selected_expert_materialization_time_this_run"] += elapsed
        self.progress["layer_timings"].setdefault(layer, {})
        self.progress["layer_timings"][layer].setdefault("selected_experts", {})[str(expert_id)] = {
            "seconds": elapsed,
            "packed_bytes_read": packed_bytes,
            "mode": "direct_slice",
        }
        return expert_state

    def load_qwen35_expert_direct(self, shard_name: str) -> Dict[str, torch.Tensor]:
        layer, expert_id, expert_prefix = self._parse_expert_shard_name(shard_name)
        started = time.perf_counter()
        expert_state = self._slice_qwen35_expert_state(layer, expert_id, prefixed=False)
        elapsed = time.perf_counter() - started
        packed_bytes = sum(t.numel() * t.element_size() for t in expert_state.values())
        self._mark_expert_ready(expert_prefix, selected_for_current_run=True, newly_built=True, direct_slice=True)
        self.progress["direct_slice_expert_loads_this_run"] += 1
        self.progress["direct_slice_packed_bytes_read_this_run"] += packed_bytes
        self.progress["selected_expert_materialization_time_this_run"] += elapsed
        self.progress["layer_timings"].setdefault(layer, {})
        self.progress["layer_timings"][layer].setdefault("selected_experts", {})[str(expert_id)] = {
            "seconds": elapsed,
            "packed_bytes_read": packed_bytes,
            "mode": "direct_slice",
        }
        return expert_state

    def load_qwen35_experts_direct(self, shard_names: Iterable[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        parsed = [self._parse_expert_shard_name(shard_name) for shard_name in shard_names]
        if not parsed:
            return {}
        layers = {layer for layer, _, _ in parsed}
        if len(layers) != 1:
            raise ValueError("Batched Qwen direct_slice requires all selected experts to come from one layer.")

        layer = parsed[0][0]
        expert_ids = [expert_id for _, expert_id, _ in parsed]
        prefixes = {expert_id: expert_prefix for _, expert_id, expert_prefix in parsed}
        started = time.perf_counter()
        expert_states = self._slice_qwen35_expert_states(layer, expert_ids, prefixed=False)
        elapsed = time.perf_counter() - started

        self.progress["layer_timings"].setdefault(layer, {})
        per_expert_elapsed = elapsed / max(1, len(expert_states))
        for expert_id, expert_state in expert_states.items():
            expert_prefix = prefixes[int(expert_id)]
            packed_bytes = sum(t.numel() * t.element_size() for t in expert_state.values())
            self._mark_expert_ready(expert_prefix, selected_for_current_run=True, newly_built=True, direct_slice=True)
            self.progress["direct_slice_expert_loads_this_run"] += 1
            self.progress["direct_slice_packed_bytes_read_this_run"] += packed_bytes
            self.progress["selected_expert_materialization_time_this_run"] += per_expert_elapsed
            self.progress["layer_timings"][layer].setdefault("selected_experts", {})[str(expert_id)] = {
                "seconds": per_expert_elapsed,
                "packed_bytes_read": packed_bytes,
                "mode": "direct_slice_batch",
            }
        return {prefixes[int(expert_id)]: state for expert_id, state in expert_states.items()}

    def load_gpt_oss_experts_direct(self, shard_names: Iterable[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        parsed = [self._parse_expert_shard_name(shard_name) for shard_name in shard_names]
        if not parsed:
            return {}
        layers = {layer for layer, _, _ in parsed}
        if len(layers) != 1:
            raise ValueError("Batched GPT-OSS direct_slice requires all selected experts to come from one layer.")

        layer = parsed[0][0]
        expert_ids = [expert_id for _, expert_id, _ in parsed]
        prefixes = {expert_id: expert_prefix for _, expert_id, expert_prefix in parsed}
        started = time.perf_counter()
        expert_states = self._slice_gpt_oss_expert_states(layer, expert_ids, prefixed=False)
        elapsed = time.perf_counter() - started

        self.progress["layer_timings"].setdefault(layer, {})
        per_expert_elapsed = elapsed / max(1, len(expert_states))
        for expert_id, expert_state in expert_states.items():
            expert_prefix = prefixes[int(expert_id)]
            packed_bytes = sum(t.numel() * t.element_size() for t in expert_state.values())
            self._mark_expert_ready(expert_prefix, selected_for_current_run=True, newly_built=True, direct_slice=True)
            self.progress["direct_slice_expert_loads_this_run"] += 1
            self.progress["direct_slice_packed_bytes_read_this_run"] += packed_bytes
            self.progress["selected_expert_materialization_time_this_run"] += per_expert_elapsed
            self.progress["layer_timings"][layer].setdefault("selected_experts", {})[str(expert_id)] = {
                "seconds": per_expert_elapsed,
                "packed_bytes_read": packed_bytes,
                "mode": "direct_slice_batch",
            }
        return {prefixes[int(expert_id)]: state for expert_id, state in expert_states.items()}

    def flush_progress(self) -> Dict:
        self._write_progress()
        self._write_moe_manifest()
        return self.status()

    def _mark_expert_ready(self, expert_prefix, *, selected_for_current_run, newly_built, direct_slice=False):
        if not direct_slice and expert_prefix not in self.progress["completed_experts"]:
            self.progress["completed_experts"].append(expert_prefix)
        if expert_prefix not in self.progress["experts_ready_total"]:
            self.progress["experts_ready_total"].append(expert_prefix)
        if selected_for_current_run and expert_prefix not in self.progress["experts_ready_selected_for_current_run"]:
            self.progress["experts_ready_selected_for_current_run"].append(expert_prefix)
        if not selected_for_current_run and expert_prefix not in self.progress["experts_ready_background"]:
            self.progress["experts_ready_background"].append(expert_prefix)
        if newly_built:
            if selected_for_current_run:
                self.progress["selected_expert_builds_this_run"] += 1
            else:
                self.progress["unused_expert_builds_this_run"] += 1
                if expert_prefix not in self.progress["experts_unused_built"]:
                    self.progress["experts_unused_built"].append(expert_prefix)

    def _slice_gpt_oss_expert_state(self, layer: str, expert_id: int, *, prefixed: bool) -> Dict[str, torch.Tensor]:
        return self._slice_gpt_oss_expert_states(layer, [expert_id], prefixed=prefixed)[int(expert_id)]

    def _slice_selected_expert_state(self, layer: str, expert_id: int, *, prefixed: bool) -> Dict[str, torch.Tensor]:
        if self.adapter_name == "qwen3_5_moe":
            return self._slice_qwen35_expert_state(layer, expert_id, prefixed=prefixed)
        return self._slice_gpt_oss_expert_state(layer, expert_id, prefixed=prefixed)

    def _slice_qwen35_expert_state(self, layer: str, expert_id: int, *, prefixed: bool) -> Dict[str, torch.Tensor]:
        return self._slice_qwen35_expert_states(layer, [expert_id], prefixed=prefixed)[int(expert_id)]

    def _slice_qwen35_expert_states(self, layer: str, expert_ids: Iterable[int], *, prefixed: bool) -> Dict[int, Dict[str, torch.Tensor]]:
        layer_base = layer.rstrip(".")
        expert_ids = sorted({int(expert_id) for expert_id in expert_ids})
        suffixes = ("gate_up_proj", "down_proj")
        result = {expert_id: {} for expert_id in expert_ids}
        by_shard = {}
        for suffix in suffixes:
            source_key = f"{layer_base}.mlp.experts.{suffix}"
            if source_key not in self.index:
                raise KeyError(f"Cannot direct-slice Qwen3.5/Qwen3.6 expert; missing tensor {source_key}")
            shard_path = self._ensure_source_shard(self.index[source_key], tensor_key=source_key)
            by_shard.setdefault(str(shard_path), []).append((source_key, suffix))

        for shard_path, items in by_shard.items():
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for source_key, suffix in items:
                    safe_slice = handle.get_slice(source_key)
                    shape = safe_slice.get_shape()
                    if shape[0] != int(self.num_experts or shape[0]):
                        raise RuntimeError(
                            f"Qwen expert tensor {source_key} has unexpected expert axis shape {shape}; "
                            f"expected first dimension {self.num_experts}."
                        )
                    for expert_id in expert_ids:
                        if not shape or shape[0] <= int(expert_id):
                            raise IndexError(f"Expert {expert_id} is outside tensor {source_key} shape {shape}")
                        try:
                            tensor = safe_slice[int(expert_id)].contiguous().clone()
                        except Exception as exc:
                            raise RuntimeError(
                                f"safetensors could not slice selected expert {expert_id} from {source_key}; "
                                "direct_slice is unsupported for this tensor layout."
                            ) from exc
                        target_key = f"{layer_base}.mlp.experts.{int(expert_id)}.{suffix}.weight" if prefixed else f"{suffix}.weight"
                        result[int(expert_id)][target_key] = tensor
        return result

    def _slice_gpt_oss_expert_states(self, layer: str, expert_ids: Iterable[int], *, prefixed: bool) -> Dict[int, Dict[str, torch.Tensor]]:
        layer_base = layer.rstrip(".")
        expert_ids = sorted({int(expert_id) for expert_id in expert_ids})
        suffixes = (
            "gate_up_proj_blocks",
            "gate_up_proj_scales",
            "gate_up_proj_bias",
            "down_proj_blocks",
            "down_proj_scales",
            "down_proj_bias",
        )
        result = {expert_id: {} for expert_id in expert_ids}
        by_shard = {}
        for suffix in suffixes:
            source_key = f"{layer_base}.mlp.experts.{suffix}"
            if source_key not in self.index:
                raise KeyError(f"Cannot direct-slice GPT-OSS expert; missing tensor {source_key}")
            shard_path = self._ensure_source_shard(self.index[source_key], tensor_key=source_key)
            by_shard.setdefault(str(shard_path), []).append((source_key, suffix))

        for shard_path, items in by_shard.items():
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for source_key, suffix in items:
                    safe_slice = handle.get_slice(source_key)
                    shape = safe_slice.get_shape()
                    for expert_id in expert_ids:
                        if not shape or shape[0] <= expert_id:
                            raise IndexError(f"Expert {expert_id} is outside tensor {source_key} shape {shape}")
                        target_key = f"{layer_base}.mlp.experts.{expert_id}.{suffix}" if prefixed else suffix
                        try:
                            result[expert_id][target_key] = safe_slice[expert_id].contiguous().clone()
                        except Exception as exc:
                            raise RuntimeError(
                                f"safetensors could not slice selected expert {expert_id} from {source_key}; "
                                "direct_slice is unsupported for this tensor layout."
                            ) from exc
        return result

    def build_until_ready(self, max_layers: Optional[int] = None):
        built = 0
        for layer in self.layers:
            if max_layers is not None and built >= max_layers:
                break
            if self._valid_saved(layer):
                self.ensure_layer(layer)
                continue
            self.build_layer(layer)
            built += 1
        return self.status()

    def build_layer(self, layer: str) -> Dict:
        started = time.perf_counter()
        layer_state_dict = {}
        bytes_read = 0
        for shard in self._layer_shards(layer):
            shard_path = self._ensure_source_shard(shard, tensor_key=layer)
            if shard_path.suffix != ".safetensors":
                raise NotImplementedError(
                    "Incremental MoE split building currently requires safetensors checkpoints."
                )
            shard_layer_keys = []
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                shard_layer_keys = [key for key in handle.keys() if key.startswith(layer)]
                for key in shard_layer_keys:
                    tensor = handle.get_tensor(key)
                    layer_state_dict[key] = tensor
                    bytes_read += tensor.numel() * tensor.element_size()

        dense_state_dict, expert_state_dicts, metadata = split_moe_layer_state_dict(
            layer_state_dict,
            num_experts=self.num_experts,
            return_metadata=True,
            drop_fused_from_dense=bool(self.adapter_name),
        )
        if metadata["fused_tensor_keys"] and not self.adapter_name:
            if self.strict_mode or not self.allow_dense_fallback:
                raise NotImplementedError("Fused expert tensors require a selective adapter in strict incremental split mode.")
        if any(".experts." in key for key in dense_state_dict):
            raise RuntimeError(f"Dense shard for {layer} contains fused expert tensors.")

        write_started = time.perf_counter()
        dense_bytes = self._save_state(dense_state_dict, layer)
        write_elapsed = time.perf_counter() - write_started
        expert_bytes = 0
        expert_started = time.perf_counter()
        for expert_prefix, expert_state in expert_state_dicts.items():
            newly_built = False
            if not self._valid_saved(expert_prefix + "."):
                expert_bytes += self._save_state(expert_state, expert_prefix + ".")
                newly_built = True
            self._mark_expert_ready(
                expert_prefix,
                selected_for_current_run=False,
                newly_built=newly_built,
            )
        expert_elapsed = time.perf_counter() - expert_started

        if layer not in self.progress["completed_layers"]:
            self.progress["completed_layers"].append(layer)
        if layer not in self.progress["dense_layers_ready"]:
            self.progress["dense_layers_ready"].append(layer)
        if layer not in self.progress["completed_dense_shards"]:
            self.progress["completed_dense_shards"].append(layer)
        self.progress["bytes_processed"] += bytes_read
        elapsed = time.perf_counter() - started
        self.progress["layer_timings"][layer] = {
            "seconds": elapsed,
            "bytes_read": bytes_read,
            "dense_bytes_written": dense_bytes,
            "expert_bytes_written": expert_bytes,
            "bytes_per_second": bytes_read / elapsed if elapsed > 0 else 0,
            "expert_extraction_seconds": expert_elapsed,
            "safetensors_write_seconds": write_elapsed + expert_elapsed,
        }
        self._write_progress()
        self._write_moe_manifest()
        return self.progress["layer_timings"][layer]

    def _write_moe_manifest(self):
        with self._progress_lock:
            experts = sorted(self.progress.get("completed_experts", []))
            dense_layers_ready = sorted(self.progress.get("dense_layers_ready", []))
            payload = {
                "moe_layout": self.adapter_name or "fused",
                "has_fused_experts": bool(self.num_experts),
                "fused_experts_split": bool(self.adapter_name or experts),
                "expert_layer_names": experts,
                "expert_count": len(experts),
                "direct_expert_layer_names": [],
                "direct_expert_count": 0,
                "fused_expert_layer_names": experts,
                "fused_expert_count": len(experts),
                "fused_tensor_names": [],
                "fused_tensor_count": 0,
                "num_experts_from_config": self.num_experts,
                "selective_fused_runtime": bool(self.adapter_name),
                "dense_contains_full_fused_experts": False,
                "requires_dense_fallback": not bool(self.adapter_name),
                "adapter_name": self.adapter_name,
                "mxfp4_execution": "reference_dequant" if self.adapter_name == "gpt_oss_mxfp4_reference" else None,
                "shared_expert_remains_dense": self.adapter_name == "qwen3_5_moe",
                "incremental_split": True,
                "partial_ready": True,
                "dense_layers_ready": dense_layers_ready,
                "dense_layers_ready_count": len(dense_layers_ready),
                "all_experts_ready": bool(self.num_experts and len(experts) >= int(self.progress.get("total_experts") or 0)),
            }
            _json_write_atomic(self.moe_manifest_path, payload)

    def verify_layer(self, layer_name: str) -> Dict:
        layer = layer_name if layer_name.endswith(".") else layer_name + "."
        if not self._valid_saved(layer):
            raise FileNotFoundError(f"Missing or invalid dense shard: {layer}")
        with safe_open(str(self._shard_path(layer)), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys:
                raise RuntimeError(f"Dense shard {layer} has no tensors.")
            bad = [key for key in keys if ".experts." in key]
            if bad:
                raise RuntimeError(f"Dense shard {layer} contains expert tensors: {bad[:4]}")
            tensor_metadata = {
                key: {
                    "shape": list(handle.get_slice(key).get_shape()),
                    "dtype": str(handle.get_tensor(key).dtype),
                }
                for key in keys
            }
        return {
            "layer": layer,
            "valid": True,
            "path": str(self._shard_path(layer)),
            "size_bytes": self._shard_path(layer).stat().st_size,
            "tensor_count": len(keys),
            "tensors": tensor_metadata,
        }

    def status(self) -> Dict:
        completed_layers = set(self.progress.get("completed_layers", []))
        completed_experts = set(self.progress.get("completed_experts", []))
        total_layers = len(self.layers)
        total_experts = int(self.progress.get("total_experts") or 0)
        elapsed = max(0.001, _now() - float(self.progress.get("started_at", _now())))
        processed = int(self.progress.get("bytes_processed", 0))
        remaining = int(self.progress.get("estimated_remaining_bytes", 0))
        bps = processed / elapsed
        eta = remaining / bps if bps > 0 else None
        return {
            "split_dir": str(self.saving_path),
            "total_layers": total_layers,
            "completed_layers": len(completed_layers),
            "dense_layers_ready": len(set(self.progress.get("dense_layers_ready", []))),
            "router_layers_ready": len(set(self.progress.get("router_layers_ready", []))),
            "total_experts": total_experts,
            "completed_experts": len(completed_experts),
            "experts_ready_total": len(set(self.progress.get("experts_ready_total", []))),
            "experts_ready_selected_for_current_run": len(set(self.progress.get("experts_ready_selected_for_current_run", []))),
            "experts_ready_background": len(set(self.progress.get("experts_ready_background", []))),
            "experts_unused_built": len(set(self.progress.get("experts_unused_built", []))),
            "selected_expert_builds_this_run": int(self.progress.get("selected_expert_builds_this_run", 0)),
            "unused_expert_builds_this_run": int(self.progress.get("unused_expert_builds_this_run", 0)),
            "direct_slice_expert_loads_this_run": int(self.progress.get("direct_slice_expert_loads_this_run", 0)),
            "direct_slice_packed_bytes_read_this_run": int(self.progress.get("direct_slice_packed_bytes_read_this_run", 0)),
            "selected_expert_materialization_time_this_run": float(self.progress.get("selected_expert_materialization_time_this_run", 0.0)),
            "progress_percent": (len(completed_layers) / total_layers * 100) if total_layers else 0,
            "bytes_processed": processed,
            "bytes_per_second": bps,
            "estimated_remaining_bytes": remaining,
            "estimated_remaining_seconds": eta,
            "strict_mode": self.strict_mode,
            "adapter_name": self.adapter_name,
            "manifest_path": str(self.manifest_path),
            "progress_path": str(self.progress_path),
        }
