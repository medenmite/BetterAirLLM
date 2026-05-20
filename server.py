"""BetterAirLLM OpenAI-Compatible API Server

Wraps BetterAirLLM's layer-wise inference engine as an OpenAI-compatible
REST API so Open WebUI (or any OpenAI client) can use it.

Endpoints:
  GET  /v1/models                → list available models
  POST /v1/chat/completions      → chat completions (streaming + non-streaming)

Usage:
  python server.py                              # defaults
  AIRLLM_MODEL=Qwen/Qwen3-30B-A3B python server.py   # custom model
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
import warnings
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoConfig, TextIteratorStreamer

# ── Ensure the local airllm package is importable ─────────────
sys.path.insert(0, "air_llm")
from airllm import AutoModel  # noqa: E402

from server_config import ServerConfig, ModelEntry, load_config  # noqa: E402
from ollama_registry import (  # noqa: E402
    OllamaHTTPError,
    OllamaUnavailable,
    discover_ollama_models,
    get_ollama_discovery_cache_status,
    get_ollama_version,
    inspect_local_ollama_store,
    list_running_ollama_models,
    raw_ollama_model_name,
    proxy_ollama_chat_completion,
    stream_ollama_chat_completion,
)

try:
    from scripts.run_gpt_oss_streaming import _clean_generated_text
except Exception:  # noqa: BLE001 - server can still run for non-GPT-OSS models.
    _clean_generated_text = None

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("airllm-server")

# ── Global State ──────────────────────────────────────────────
config: ServerConfig = None  # type: ignore
loaded_model = None
loaded_model_id: Optional[str] = None
model_lock = Lock()
model_manager = None
last_generation_stats: dict[str, Any] = {}
runtime_lock = Lock()


# ═════════════════════════════════════════════════════════════
# Pydantic Models (OpenAI API Schema)
# ═════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str
    content: Union[str, list[dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str = "airllm"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    reasoning_effort: Optional[str] = None
    reasoning: Optional[dict[str, Any]] = None
    # Accept but ignore these common OpenAI params
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, list[str]]] = None


# ═════════════════════════════════════════════════════════════
# Model Management
# ═════════════════════════════════════════════════════════════

async def available_model_entries() -> list[ModelEntry]:
    """Return configured BetterAirLLM/local models plus dynamically discovered Ollama models."""
    entries = list(config.models)
    if not getattr(config, "discover_ollama", True):
        return entries
    try:
        entries.extend(
            await discover_ollama_models(
                config.ollama_base_url,
                timeout_seconds=config.ollama_timeout_seconds,
                ttl_seconds=config.ollama_discovery_ttl_seconds,
            )
        )
    except OllamaUnavailable as e:
        status = inspect_local_ollama_store()
        if status.warning:
            log.warning("%s base_url=%s error=%s", status.warning, config.ollama_base_url, e)
        else:
            log.debug("Ollama discovery unavailable at %s: %s", config.ollama_base_url, e)
    return _dedupe_model_entries(entries)


def _dedupe_model_entries(entries: list[ModelEntry]) -> list[ModelEntry]:
    seen = set()
    deduped = []
    for entry in entries:
        if entry.id in seen:
            continue
        seen.add(entry.id)
        deduped.append(entry)
    return deduped


def find_config_model_entry(model_id: str) -> Optional[ModelEntry]:
    """Look up a model by ID in the registry."""
    for m in config.models:
        if m.id == model_id:
            return m
    # Fallback: try matching by repo_id
    for m in config.models:
        if m.repo_id == model_id or m.repo_id.split("/")[-1].lower() == model_id.lower():
            return m
    return None


async def find_model_entry(model_id: str) -> Optional[ModelEntry]:
    configured = find_config_model_entry(model_id)
    if configured is not None:
        return configured
    if model_id.startswith("ollama/"):
        raw_ollama_name = raw_ollama_model_name(model_id)
        for entry in await available_model_entries():
            if entry.ollama_model == raw_ollama_name:
                return entry
        return None
    if len(config.models) == 1:
        return config.models[0]

    if getattr(config, "discover_ollama", True):
        raw_ollama_name = raw_ollama_model_name(model_id)
        for entry in await available_model_entries():
            if entry.id == model_id or entry.ollama_model == raw_ollama_name:
                return entry
    return None


class ModelManager:
    """Small LRU manager for loaded BetterAirLLM models."""

    def __init__(self, max_loaded_models: int = 1) -> None:
        self.max_loaded_models = max(1, int(max_loaded_models))
        self._models: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = model_lock
        self.loads = 0
        self.reuses = 0
        self.evictions = 0
        self.last_error: Optional[str] = None

    def get_or_load_model(self, model_id: str):
        entry = find_config_model_entry(model_id)
        if entry is None:
            if len(config.models) == 1:
                entry = config.models[0]
            else:
                raise ValueError(f"Model '{model_id}' not found in registry. Available: {[m.id for m in config.models]}")

        if entry.backend == "ollama":
            raise ValueError(f"Model '{entry.id}' uses the Ollama backend and should be proxied, not loaded by BetterAirLLM.")
        if str(entry.format or "").lower() == "gguf":
            raise ValueError("GGUF cannot currently run through BetterAirLLM; use backend=ollama or provide a HF/Safetensors checkpoint.")

        with self._lock:
            if entry.id in self._models:
                record = self._models.pop(entry.id)
                record["last_used_at"] = time.time()
                self._models[entry.id] = record
                self.reuses += 1
                _sync_legacy_loaded_model(record["model"], entry)
                log.info("Model '%s' already loaded, reusing.", entry.id)
                return record["model"], entry

            while len(self._models) >= self.max_loaded_models:
                evicted_id, evicted = self._models.popitem(last=False)
                log.info("Unloading LRU model '%s'...", evicted_id)
                del evicted["model"]
                self.evictions += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            log.info("Loading model '%s' from '%s'...", entry.id, entry.repo_id)
            log.info("  device=%s, compression=%s, max_seq_len=%s", config.device, entry.compression, entry.max_seq_len)
            kwargs = _build_model_load_kwargs(entry)
            try:
                model = AutoModel.from_pretrained(entry.repo_id, **kwargs)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise

            self.loads += 1
            self.last_error = None
            self._models[entry.id] = {
                "model": model,
                "entry": entry,
                "loaded_at": time.time(),
                "last_used_at": time.time(),
                "load_kwargs": _safe_load_kwargs_for_report(kwargs),
            }
            _sync_legacy_loaded_model(model, entry)
            log.info("[OK] Model '%s' loaded successfully.", entry.id)
            return model, entry

    def runtime_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_loaded_models": self.max_loaded_models,
                "loaded_model_ids": list(self._models.keys()),
                "loaded_models": [
                    {
                        "id": model_id,
                        "backend": record["entry"].backend,
                        "repo_id": record["entry"].repo_id,
                        "loaded_at": record["loaded_at"],
                        "last_used_at": record["last_used_at"],
                        "load_kwargs": record.get("load_kwargs", {}),
                    }
                    for model_id, record in self._models.items()
                ],
                "loads": self.loads,
                "reuses": self.reuses,
                "evictions": self.evictions,
                "last_error": self.last_error,
            }


def _get_model_manager() -> ModelManager:
    global model_manager
    desired_max = max(1, int(getattr(config, "max_loaded_models", 1)))
    if model_manager is None or getattr(model_manager, "max_loaded_models", None) != desired_max:
        model_manager = ModelManager(desired_max)
    return model_manager


def _sync_legacy_loaded_model(model, entry: ModelEntry) -> None:
    global loaded_model, loaded_model_id
    loaded_model = model
    loaded_model_id = entry.id


def get_or_load_model(model_id: str):
    """Load a model if not already loaded. Thread-safe via ModelManager."""

    return _get_model_manager().get_or_load_model(model_id)


def _build_model_load_kwargs(entry: ModelEntry) -> dict[str, Any]:
    kwargs = {
        "device": config.device,
        "max_seq_len": entry.max_seq_len,
        "prefetching": config.prefetching,
    }
    if entry.compression:
        kwargs["compression"] = entry.compression
    if config.hf_token:
        kwargs["hf_token"] = config.hf_token

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    kwargs["dtype"] = dtype_map.get(config.dtype, torch.float16)

    if _is_gpt_oss_entry(entry):
        kwargs.update({
            "dense_device": config.dense_device,
            "moe_strict_streaming": True,
            "moe_allow_dense_fallback": False,
            "moe_expert_cache_mb": 1024,
            "moe_cpu_expert_cache_mb": config.cpu_cache_mb,
            "vram_cache_mode": config.vram_cache_mode,
            "vram_cache_mb": config.vram_cache_mb,
            "os_reserved_ram_mb": config.os_reserved_ram_mb,
            "abort_unsafe_context": True,
            "resume_split_build": True,
            "lazy_build": True,
            "expert_materialization_mode": "direct_slice",
            "enable_qwen35_batch_direct_slice": True,
            "dense_expert_cache_min_requests": 1,
            "layer_cleanup_interval": config.layer_cleanup_interval,
            "mxfp4_device": config.mxfp4_device,
            "expert_matmul_device": config.expert_matmul_device,
            "mxfp4_execution": _resolved_mxfp4_execution(),
            "hf_triton_module_cache_mb": config.hf_triton_module_cache_mb,
            "max_vram_mb": config.max_vram_mb,
            "enable_gpt_oss_batch_direct_slice": config.expert_execution_mode == "grouped_by_expert",
            "expert_execution_mode": config.expert_execution_mode,
            "keep_nontransformer_resident": True,
            "quiet_progress": config.quiet_progress,
        })
    elif _is_qwen35_entry(entry):
        kwargs.update({
            "dense_device": config.dense_device,
            "moe_strict_streaming": True,
            "moe_allow_dense_fallback": False,
            "moe_expert_cache_mb": 1024,
            "moe_cpu_expert_cache_mb": config.cpu_cache_mb,
            "vram_cache_mode": config.vram_cache_mode,
            "vram_cache_mb": config.vram_cache_mb,
            "os_reserved_ram_mb": config.os_reserved_ram_mb,
            "abort_unsafe_context": True,
            "resume_split_build": True,
            "lazy_build": True,
            "expert_materialization_mode": "direct_slice",
            "layer_cleanup_interval": config.layer_cleanup_interval,
            "expert_execution_mode": config.expert_execution_mode,
            "max_vram_mb": config.max_vram_mb,
            "keep_nontransformer_resident": True,
            "quiet_progress": config.quiet_progress,
        })
    return kwargs


def _safe_load_kwargs_for_report(kwargs: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in kwargs.items():
        if key == "hf_token":
            safe[key] = "configured"
        elif isinstance(value, torch.dtype):
            safe[key] = str(value).replace("torch.", "")
        else:
            safe[key] = value
    return safe


def _resolved_mxfp4_execution() -> str:
    requested = str(getattr(config, "mxfp4_execution", "reference_cuda") or "reference_cuda").lower()
    if requested == "triton_fused":
        requested = "hf_triton"
    if requested != "hf_triton":
        return requested
    status = _hf_triton_status()
    if status["available"]:
        return "hf_triton"
    log.warning("HF Triton MXFP4 execution unavailable (%s); using reference_cuda.", status["reason"])
    return "reference_cuda"


def _is_gpt_oss_entry(entry: ModelEntry) -> bool:
    marker = f"{entry.id} {entry.repo_id}".lower()
    return "gpt-oss" in marker or "gpt_oss" in marker


def _is_qwen35_entry(entry: ModelEntry) -> bool:
    marker = f"{entry.id} {entry.repo_id}".lower()
    return (
        "qwen3.6-35b-a3b" in marker
        or "qwen3-6-35b-a3b" in marker
        or "qwen3_5_moe" in marker
        or "qwen3.5-35b-a3b" in marker
        or "qwen3-5-35b-a3b" in marker
    )


def _hf_triton_status() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda_unavailable", "cuda_available": False}
    try:
        import transformers.integrations.mxfp4  # noqa: F401
        from transformers.integrations.hub_kernels import get_kernel  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "cuda_available": True,
        }
    return {"available": True, "reason": "available", "cuda_available": True}


def _hardware_profile() -> dict[str, Any]:
    profile: dict[str, Any] = {
        "device": config.device,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": getattr(torch, "__version__", None),
    }
    if torch.cuda.is_available():
        try:
            index = _cuda_device_index(config.device)
            props = torch.cuda.get_device_properties(index)
            profile["cuda"] = {
                "device_index": index,
                "name": torch.cuda.get_device_name(index),
                "total_vram_mb": int(getattr(props, "total_memory", 0) / 1024**2),
                "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(index)),
                "allocated_mb": int(torch.cuda.memory_allocated(index) / 1024**2),
                "reserved_mb": int(torch.cuda.memory_reserved(index) / 1024**2),
                "max_allocated_mb": int(torch.cuda.max_memory_allocated(index) / 1024**2),
                "max_reserved_mb": int(torch.cuda.max_memory_reserved(index) / 1024**2),
            }
        except Exception as exc:
            profile["cuda_error"] = f"{type(exc).__name__}: {exc}"
    try:
        usage = shutil.disk_usage(Path.cwd())
        profile["workspace_disk"] = {
            "path": str(Path.cwd()),
            "total_mb": int(usage.total / 1024**2),
            "free_mb": int(usage.free / 1024**2),
            "used_mb": int(usage.used / 1024**2),
        }
    except Exception:
        pass
    return profile


def _model_support_payload(entry: ModelEntry) -> dict[str, Any]:
    metadata = dict(entry.metadata or {})
    family = entry.family or metadata.get("family") or _infer_model_family(entry)
    status = entry.status or metadata.get("status") or ("experimental" if family in {"moe", "qwen_moe", "qwen3_5_moe", "gpt_oss"} else "supported")
    tested_level = entry.tested_level or metadata.get("tested_level") or ("architecture_dispatch" if entry.backend == "airllm" else "backend_proxy")
    return {
        "family": family,
        "prompt_format": entry.prompt_format or metadata.get("prompt_format") or "tokenizer_chat_template",
        "status": status,
        "tested_level": tested_level,
        "quantization": entry.quantization or metadata.get("quantization"),
        "memory": entry.memory or metadata.get("memory") or {},
        "context_length": entry.context_length or metadata.get("context_length") or entry.max_seq_len,
        "known_limitations": entry.known_limitations or metadata.get("known_limitations") or [],
    }


def _infer_model_family(entry: ModelEntry) -> str:
    marker = f"{entry.id} {entry.repo_id}".lower()
    if entry.backend == "ollama":
        return str((entry.metadata or {}).get("family") or "ollama")
    if "gpt-oss" in marker or "gpt_oss" in marker:
        return "gpt_oss"
    if _is_qwen35_entry(entry):
        return "qwen3_5_moe"
    if "qwen" in marker and ("moe" in marker or "a3b" in marker):
        return "qwen_moe"
    if "qwen" in marker:
        return "qwen"
    if "mistral" in marker:
        return "mistral"
    if "llama" in marker:
        return "llama"
    return "unknown_fallback"


def build_capabilities() -> dict[str, Any]:
    """Return static and configured runtime capabilities."""

    configured_airllm = [m.id for m in config.models if m.backend == "airllm"]
    configured_ollama = [m.id for m in config.models if m.backend == "ollama"]
    triton_status = _hf_triton_status()
    return {
        "service": "BetterAirLLM",
        "api_version": "1.0.0",
        "hardware": _hardware_profile(),
        "runtime": {
            "stream_mode": config.stream_mode,
            "runtime_metrics_enabled": config.enable_runtime_metrics,
            "max_loaded_models": config.max_loaded_models,
            "mxfp4_execution_configured": config.mxfp4_execution,
            "mxfp4_execution_effective": _resolved_mxfp4_execution() if triton_status["available"] else "reference_cuda",
            "hf_triton": triton_status,
            "hf_triton_module_cache_mb": config.hf_triton_module_cache_mb,
        },
        "backends": [
            {
                "id": "airllm",
                "description": "Hugging Face or local safetensors checkpoints through BetterAirLLM layer-wise loading.",
                "formats": ["safetensors", "pytorch_bin", "unknown"],
            },
            {
                "id": "ollama",
                "description": "Local Ollama OpenAI-compatible proxy for installed GGUF models.",
                "formats": ["gguf"],
            },
        ],
        "configured_model_count": len(config.models),
        "configured_airllm_model_count": len(configured_airllm),
        "configured_ollama_model_count": len(configured_ollama),
        "configured_airllm_models": configured_airllm,
        "configured_ollama_models": configured_ollama,
        "ollama": {
            "discovery_enabled": config.discover_ollama,
            "base_url": config.ollama_base_url,
            "discovery_ttl_seconds": config.ollama_discovery_ttl_seconds,
        },
        "hf_architecture_families": [
            {"id": "moe", "runtime_class": "BetterAirLLMMoE", "stability": "experimental"},
            {"id": "qwen2_qwen2_5", "runtime_class": "BetterAirLLMQWen2", "stability": "supported"},
            {"id": "qwen", "runtime_class": "BetterAirLLMQWen", "stability": "supported"},
            {"id": "baichuan", "runtime_class": "BetterAirLLMBaichuan", "stability": "supported"},
            {"id": "chatglm", "runtime_class": "BetterAirLLMChatGLM", "stability": "supported"},
            {"id": "internlm", "runtime_class": "BetterAirLLMInternLM", "stability": "supported"},
            {"id": "mistral", "runtime_class": "BetterAirLLMMistral", "stability": "supported"},
            {"id": "llama", "runtime_class": "BetterAirLLMLlama2", "stability": "supported"},
            {"id": "unknown_fallback", "runtime_class": "BetterAirLLMLlama2", "stability": "best_effort"},
        ],
        "experimental_features": [
            {
                "id": "gpt_oss_mxfp4_hf_triton",
                "description": "GPT-OSS MXFP4 selected-expert execution through Hugging Face Triton kernels when CUDA/kernels are available.",
                "status": "experimental",
                "available": triton_status["available"],
                "fallback": "reference_cuda" if not triton_status["available"] else None,
            },
            {
                "id": "gpt_oss_mxfp4_reference",
                "description": "GPT-OSS MXFP4 selected-expert reference execution fallback.",
                "status": "experimental",
            },
            {
                "id": "qwen3_5_qwen3_6_moe_selective_runtime",
                "description": "Qwen3.5/Qwen3.6 MoE selective routed expert streaming.",
                "status": "experimental",
            },
        ],
        "known_limitations": [
            "GGUF files are not loaded directly through the BetterAirLLM backend; use the Ollama backend.",
            "GPT-OSS HF Triton acceleration requires CUDA and compatible transformers/hub-kernels dependencies.",
            "Mixtral, generic Qwen-MoE, and DeepSeek fused selective runtime require model-specific adapters before being claimed as fast selective runtimes.",
        ],
        "future_work": [
            "Implement a BetterAirLLM-owned fused MXFP4 kernel if the HF Triton path is insufficient.",
            "Broaden model-specific MoE adapter correctness coverage.",
            "Add live desktop chat integration and model selection in the Next.js pass.",
        ],
    }


def build_model_preflight(entry: ModelEntry, *, requested_model_id: Optional[str] = None) -> dict[str, Any]:
    """Build a non-loading model readiness report."""

    warnings_out: list[str] = []
    blockers: list[str] = []
    if entry.backend == "ollama":
        return _build_ollama_preflight(entry, requested_model_id=requested_model_id)

    if str(entry.format or "").lower() == "gguf":
        blockers.append("GGUF cannot run through the BetterAirLLM backend. Use backend=ollama.")

    marker = f"{entry.id} {entry.repo_id}".lower()
    if ("meta-llama" in marker or "llama-2" in marker) and not config.hf_token:
        warnings_out.append("This model may require HF_TOKEN for gated Hugging Face access.")

    device_report = _device_preflight_report(config.device)
    if str(config.device).startswith("cuda") and not device_report.get("cuda_available"):
        blockers.append(f"Configured device {config.device} requests CUDA, but torch.cuda.is_available() is false.")

    triton_status = _hf_triton_status()
    if _is_gpt_oss_entry(entry) and str(config.mxfp4_execution).lower() in {"hf_triton", "triton_fused"} and not triton_status["available"]:
        warnings_out.append(f"HF Triton MXFP4 acceleration unavailable; reference_cuda will be used. Reason: {triton_status['reason']}")

    config_summary = _local_hf_config_summary(entry.repo_id, config.hf_token)
    warnings_out.extend(config_summary.pop("warnings", []))
    paths = _split_path_hints(entry.repo_id)

    return {
        "status": "blocked" if blockers else ("warning" if warnings_out else "ok"),
        "model_id": entry.id,
        "requested_model_id": requested_model_id or entry.id,
        "registry_entry": _model_entry_payload(entry),
        "support": _model_support_payload(entry),
        "backend": entry.backend,
        "source": entry.source,
        "format": entry.format,
        "max_seq_len": entry.max_seq_len,
        "runtime": {
            "stream_mode": config.stream_mode,
            "mxfp4_execution_configured": config.mxfp4_execution,
            "mxfp4_execution_effective": _resolved_mxfp4_execution() if _is_gpt_oss_entry(entry) else None,
            "hf_triton": triton_status,
            "hf_triton_module_cache_mb": config.hf_triton_module_cache_mb,
            "lazy_build": _is_gpt_oss_entry(entry) or _is_qwen35_entry(entry),
            "resume_split_build": _is_gpt_oss_entry(entry) or _is_qwen35_entry(entry),
        },
        "hardware": _hardware_profile(),
        "device": device_report,
        "hf": {
            "token_configured": bool(config.hf_token),
            "config": config_summary,
        },
        "paths": paths,
        "warnings": warnings_out,
        "blockers": blockers,
    }


def _build_ollama_preflight(entry: ModelEntry, *, requested_model_id: Optional[str] = None) -> dict[str, Any]:
    raw_name = entry.ollama_model or raw_ollama_model_name(requested_model_id or entry.id)
    cache_status = get_ollama_discovery_cache_status(config.ollama_base_url)
    local_status = inspect_local_ollama_store()
    metadata = dict(entry.metadata or {})
    warnings_out = []
    blockers = []
    if cache_status.last_discovery_error:
        warnings_out.append(f"Last Ollama discovery error: {cache_status.last_discovery_error}")
    if local_status.warning:
        warnings_out.append(local_status.warning)
    if not raw_name:
        blockers.append("No raw Ollama model name is available for this entry.")
    return {
        "status": "blocked" if blockers else ("warning" if warnings_out else "ok"),
        "model_id": entry.id,
        "requested_model_id": requested_model_id or entry.id,
        "registry_entry": _model_entry_payload(entry),
        "support": _model_support_payload(entry),
        "backend": "ollama",
        "source": entry.source,
        "format": entry.format,
        "ollama": {
            "base_url": config.ollama_base_url,
            "discovery_enabled": config.discover_ollama,
            "daemon_status_from_cache": "available" if cache_status.has_cache else "unknown",
            "raw_model": raw_name,
            "discoverable": True,
            "context_length": metadata.get("context_length"),
            "family": metadata.get("family"),
            "parameter_size": metadata.get("parameter_size"),
            "quantization_level": metadata.get("quantization_level"),
            "size": metadata.get("size"),
            "cache_age_seconds": cache_status.cache_age_seconds,
            "models_path": local_status.models_path,
            "models_path_exists": local_status.models_path_exists,
            "blobs_found": local_status.blobs_found,
        },
        "metadata": metadata,
        "warnings": warnings_out,
        "blockers": blockers,
    }


async def build_missing_model_preflight(model_id: str) -> dict[str, Any]:
    blockers = [f"Model '{model_id}' is not configured or discoverable."]
    payload: dict[str, Any] = {
        "status": "blocked",
        "model_id": model_id,
        "requested_model_id": model_id,
        "warnings": [],
        "blockers": blockers,
    }
    if model_id.startswith("ollama/"):
        payload["ollama"] = {
            "detail": await _ollama_model_not_found_detail(model_id),
            "base_url": config.ollama_base_url,
            "discovery_enabled": config.discover_ollama,
        }
    else:
        payload["available_models"] = [m.id for m in await available_model_entries()]
    return payload


def _model_entry_payload(entry: ModelEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "repo_id": entry.repo_id,
        "description": entry.description,
        "compression": entry.compression,
        "max_seq_len": entry.max_seq_len,
        "owned_by": entry.owned_by,
        "source": entry.source,
        "backend": entry.backend,
        "format": entry.format,
        "ollama_model": entry.ollama_model,
        "family": entry.family,
        "prompt_format": entry.prompt_format,
        "status": entry.status,
        "tested_level": entry.tested_level,
        "quantization": entry.quantization,
        "memory": entry.memory,
        "context_length": entry.context_length,
        "known_limitations": entry.known_limitations,
        "metadata": entry.metadata,
    }


def _device_preflight_report(device: str) -> dict[str, Any]:
    report: dict[str, Any] = {"requested": device, "cuda_requested": str(device).startswith("cuda")}
    if not str(device).startswith("cuda"):
        return report
    try:
        cuda_available = torch.cuda.is_available()
        report["cuda_available"] = cuda_available
        report["torch_cuda_version"] = torch.version.cuda
        if not cuda_available:
            return report
        device_index = _cuda_device_index(device)
        torch_device = torch.device(device)
        props = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch_device)
        capability = torch.cuda.get_device_capability(device_index)
        report.update({
            "index": device_index,
            "name": torch.cuda.get_device_name(device_index),
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "total_vram_bytes": int(total_bytes),
            "free_vram_bytes": int(free_bytes),
            "configured_max_vram_mb": config.max_vram_mb,
            "arch_list": list(getattr(torch.cuda, "get_arch_list", lambda: [])()),
        })
    except Exception as exc:  # noqa: BLE001
        report["warning"] = str(exc)
    return report


def _local_hf_config_summary(repo_id: str, hf_token: Optional[str]) -> dict[str, Any]:
    try:
        kwargs = {"trust_remote_code": True, "local_files_only": True}
        if hf_token:
            kwargs["token"] = hf_token
        model_config = AutoConfig.from_pretrained(repo_id, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "available_locally": False,
            "warnings": [f"Hugging Face config is not available locally without download: {exc}"],
        }

    architectures = getattr(model_config, "architectures", None) or []
    quantization = getattr(model_config, "quantization_config", None)
    return {
        "available_locally": True,
        "model_type": getattr(model_config, "model_type", None),
        "architectures": architectures,
        "quantization_config": quantization if isinstance(quantization, dict) else None,
    }


def _split_path_hints(repo_id: str) -> dict[str, Any]:
    path = Path(repo_id)
    if path.exists():
        base_path = path
    else:
        cache_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        repo_cache_name = f"models--{repo_id.replace('/', '--')}"
        base_path = cache_home / "hub" / repo_cache_name
    split_path = base_path / "splitted_model"
    progress_path = split_path / "progress.json"
    disk_target = base_path if base_path.exists() else base_path.parent
    disk_free_mb = None
    try:
        usage = shutil.disk_usage(disk_target if disk_target.exists() else Path.cwd())
        disk_free_mb = int(usage.free / 1024**2)
    except Exception:
        pass
    return {
        "model_path_hint": str(base_path),
        "model_path_exists": base_path.exists(),
        "split_dir_name": "splitted_model",
        "split_path_hint": str(split_path),
        "split_path_exists": split_path.exists(),
        "split_progress_path": str(progress_path),
        "split_progress_exists": progress_path.exists(),
        "disk_free_mb": disk_free_mb,
    }


def _message_content_to_text(content: Union[str, list[dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        part_type = str(part.get("type", "")).lower()
        if part_type == "text":
            parts.append(str(part.get("text", "")))
        elif part_type == "image_url":
            continue
        elif "video" in part_type:
            raise ValueError("video content parts are not supported by this v1 chat endpoint")
        else:
            raise ValueError(f"unsupported chat content part type: {part_type!r}")
    return "\n".join(text for text in parts if text).strip()


def _messages_have_image(messages: list[ChatMessage]) -> bool:
    for message in messages:
        if isinstance(message.content, list):
            if any(str(part.get("type", "")).lower() == "image_url" for part in message.content):
                return True
    return False


def _normalize_openai_messages(messages: list[ChatMessage], *, text_only: bool = False) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        content = _message_content_to_text(message.content) if text_only else message.content
        if isinstance(content, list):
            for part in content:
                part_type = str(part.get("type", "")).lower()
                if "video" in part_type:
                    raise ValueError("video content parts are not supported by this v1 chat endpoint")
                if part_type not in {"text", "image_url"}:
                    raise ValueError(f"unsupported chat content part type: {part_type!r}")
        normalized.append({"role": message.role, "content": content})
    return normalized


# ═════════════════════════════════════════════════════════════
# Chat Template
# ═════════════════════════════════════════════════════════════

def build_prompt(model, messages: list[ChatMessage], entry: ModelEntry) -> str:
    """Convert OpenAI messages to a single prompt string.

    Uses the tokenizer's chat_template if available, otherwise falls
    back to a simple format.
    """
    msgs = _normalize_openai_messages(messages, text_only=not _messages_have_image(messages))

    if _is_gpt_oss_entry(entry) and config.use_harmony_prompt:
        system_content = "You are ChatGPT, a large language model trained by OpenAI."
        user_parts = []
        for msg in messages:
            if msg.role == "system":
                system_content = _message_content_to_text(msg.content)
            elif msg.role == "user":
                user_parts.append(_message_content_to_text(msg.content))
            elif msg.role == "assistant":
                user_parts.append(f"Previous assistant answer: {_message_content_to_text(msg.content)}")
        user_content = "\n\n".join(user_parts).strip()
        msgs = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # Try native chat template first.
    try:
        if _is_qwen35_entry(entry) and _messages_have_image(messages):
            processor = getattr(model, "processor", None)
            if processor is not None and hasattr(processor, "apply_chat_template"):
                return processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        if hasattr(model.tokenizer, "apply_chat_template"):
            prompt = model.tokenizer.apply_chat_template(
                _normalize_openai_messages(messages, text_only=True),
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
    except Exception as e:
        log.warning(f"Chat template failed: {e}, using fallback format")

    # Fallback: simple format
    parts = []
    for msg in messages:
        content = _message_content_to_text(msg.content)
        if msg.role == "system":
            parts.append(f"[System]\n{content}\n")
        elif msg.role == "user":
            parts.append(f"[User]\n{content}\n")
        elif msg.role == "assistant":
            parts.append(f"[Assistant]\n{content}\n")
    parts.append("[Assistant]\n")
    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════

def run_inference(model, entry: ModelEntry, prompt: str, max_tokens: int) -> str:
    """Run BetterAirLLM inference and return generated text."""
    if getattr(config, "enable_runtime_metrics", True) and hasattr(model, "reset_runtime_stats"):
        model.reset_runtime_stats()
    input_tokens = model.tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=entry.max_seq_len,
        padding=False,
    )

    input_ids = input_tokens["input_ids"]
    input_len = input_ids.shape[1]

    if config.device.startswith("cuda"):
        input_ids = input_ids.cuda()

    log.info(f"Generating... input_len={input_len}, max_new_tokens={max_tokens}")
    t0 = time.time()

    generation_output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        use_cache=True,
        return_dict_in_generate=True,
    )

    elapsed = time.time() - t0
    output_ids = generation_output.sequences[0]
    new_tokens = len(output_ids) - input_len
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0

    log.info(f"Generated {new_tokens} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")

    # Decode only the new tokens
    output_text = model.tokenizer.decode(
        output_ids[input_len:],
        skip_special_tokens=not (_is_gpt_oss_entry(entry) and config.use_harmony_prompt),
    )
    if _is_gpt_oss_entry(entry):
        output_text = clean_output_text(output_text, prompt)
    _record_generation_stats(
        model,
        entry,
        {
            "mode": "non_streaming",
            "stream_mode": "none",
            "prompt_tokens": int(input_len),
            "completion_tokens": int(new_tokens),
            "total_tokens": int(input_len + new_tokens),
            "total_seconds": elapsed,
            "tokens_per_second": tok_per_sec,
            "time_to_first_token_seconds": None,
            "mxfp4_execution_effective": _resolved_mxfp4_execution() if _is_gpt_oss_entry(entry) else None,
        },
    )
    return output_text


def run_inference_stream(model, entry: ModelEntry, prompt: str, max_tokens: int) -> Queue:
    """Start real token streaming in a worker thread and return an event queue."""

    if str(config.stream_mode).lower() == "compat":
        raise RuntimeError("stream_mode=compat disables real streamer generation")

    event_queue: Queue = Queue()

    def worker() -> None:
        output_text = ""
        input_len = 0
        completion_tokens = 0
        first_token_at = None
        started = time.time()
        try:
            if getattr(config, "enable_runtime_metrics", True) and hasattr(model, "reset_runtime_stats"):
                model.reset_runtime_stats()
            input_tokens = model.tokenizer(
                [prompt],
                return_tensors="pt",
                return_attention_mask=False,
                truncation=True,
                max_length=entry.max_seq_len,
                padding=False,
            )
            input_ids = input_tokens["input_ids"]
            input_len = int(input_ids.shape[1])
            if config.device.startswith("cuda"):
                input_ids = input_ids.cuda()

            streamer = TextIteratorStreamer(
                model.tokenizer,
                skip_prompt=True,
                skip_special_tokens=not (_is_gpt_oss_entry(entry) and config.use_harmony_prompt),
            )
            generation_error: dict[str, BaseException] = {}

            def generate_target() -> None:
                try:
                    model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        use_cache=True,
                        streamer=streamer,
                        return_dict_in_generate=True,
                    )
                except BaseException as exc:  # noqa: BLE001
                    generation_error["error"] = exc
                    if hasattr(streamer, "on_finalized_text"):
                        streamer.on_finalized_text("", stream_end=True)

            generate_thread = Thread(target=generate_target, daemon=True)
            generate_thread.start()
            for text in streamer:
                if not text:
                    continue
                if first_token_at is None:
                    first_token_at = time.time()
                output_text += text
                completion_tokens += _token_count(model, text)
                event_queue.put(("token", clean_output_text(text, prompt) if _is_gpt_oss_entry(entry) else text))
            generate_thread.join()
            if generation_error:
                raise generation_error["error"]
            elapsed = time.time() - started
            _record_generation_stats(
                model,
                entry,
                {
                    "mode": "streaming",
                    "stream_mode": "text_iterator_streamer",
                    "prompt_tokens": input_len,
                    "completion_tokens": completion_tokens,
                    "total_tokens": input_len + completion_tokens,
                    "total_seconds": elapsed,
                    "tokens_per_second": completion_tokens / elapsed if elapsed > 0 else 0.0,
                    "time_to_first_token_seconds": (first_token_at - started) if first_token_at else None,
                    "mxfp4_execution_effective": _resolved_mxfp4_execution() if _is_gpt_oss_entry(entry) else None,
                },
            )
            event_queue.put(("done", None))
        except BaseException as exc:  # noqa: BLE001
            event_queue.put(("error", exc))

    Thread(target=worker, daemon=True).start()
    return event_queue


def _token_count(model, text: str) -> int:
    try:
        return len(model.tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"])
    except Exception:
        return max(1, len(text.split()))


def _record_generation_stats(model, entry: ModelEntry, stats: dict[str, Any]) -> None:
    if not getattr(config, "enable_runtime_metrics", True):
        return
    runtime_stats = {}
    if hasattr(model, "runtime_stats"):
        try:
            runtime_stats = model.runtime_stats()
        except Exception as exc:
            runtime_stats = {"error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "model_id": entry.id,
        "backend": entry.backend,
        "created": int(time.time()),
        **stats,
        "runtime_stats": runtime_stats,
    }
    with runtime_lock:
        last_generation_stats.clear()
        last_generation_stats.update(payload)


def guard_image_request(model, entry: ModelEntry, messages: list[ChatMessage]) -> None:
    if not _messages_have_image(messages):
        return
    if not _is_qwen35_entry(entry):
        raise HTTPException(status_code=400, detail="image_url content is only enabled for Qwen3.6/Qwen3.5 MoE entries")
    if config.max_vram_mb and int(config.max_vram_mb) < 7600:
        raise HTTPException(
            status_code=400,
            detail=f"image_url requests require AIRLLM_MAX_VRAM_MB >= 7600 for the guarded Qwen visual path; current={config.max_vram_mb}",
        )
    raise HTTPException(
        status_code=501,
        detail=(
            "Qwen image_url chat content was parsed, but BetterAirLLM does not yet stream the Qwen visual encoder. "
            "The request is rejected to avoid silently ignoring image pixels or exceeding the VRAM budget."
        ),
    )


def clean_output_text(text: str, prompt: str) -> str:
    if _clean_generated_text is not None:
        return _clean_generated_text(text, prompt)
    cleaned = re.sub(r"<\|[^|]+?\|>", "", text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(?im)^\s*(analysis|commentary|final)\s*$", "", cleaned)
    return cleaned.strip()


def _error_detail(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            **{key: value for key, value in extra.items() if value is not None},
        }
    }


def _runtime_load_http_exception(exc: RuntimeError, model_id: str) -> HTTPException:
    message = str(exc)
    lower = message.lower()
    if "cuda" in lower and ("available" in lower or "device" in lower):
        return HTTPException(status_code=503, detail=_error_detail("cuda_unavailable", message, model_id=model_id))
    if "out of memory" in lower or "vram budget" in lower:
        return HTTPException(status_code=507, detail=_error_detail("vram_or_ram_exhausted", message, model_id=model_id))
    return HTTPException(status_code=500, detail=_error_detail("model_runtime_error", message, model_id=model_id))


def _inference_http_exception(exc: Exception) -> HTTPException:
    message = str(exc)
    lower = message.lower()
    if isinstance(exc, MemoryError) or "out of memory" in lower or "vram budget" in lower:
        return HTTPException(status_code=507, detail=_error_detail("vram_or_ram_exhausted", message))
    if "cuda" in lower and "available" in lower:
        return HTTPException(status_code=503, detail=_error_detail("cuda_unavailable", message))
    return HTTPException(status_code=500, detail=_error_detail("inference_failed", message))


# ═════════════════════════════════════════════════════════════
# FastAPI App
# ═════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, model_manager
    config = load_config()
    model_manager = ModelManager(config.max_loaded_models)
    log.info(f"BetterAirLLM Server starting on {config.host}:{config.port}")
    await _log_startup_model_inventory()
    log.info(f"Device: {config.device}")
    _log_torch_cuda_status()
    yield
    log.info("Shutting down.")


async def _log_startup_model_inventory() -> None:
    configured_airllm = [m.id for m in config.models if m.backend == "airllm"]
    configured_ollama = [m.id for m in config.models if m.backend == "ollama"]

    log.info("Configured BetterAirLLM models (%d): %s", len(configured_airllm), configured_airllm)
    if configured_ollama:
        log.info("Configured Ollama models (%d): %s", len(configured_ollama), configured_ollama)

    entries = await available_model_entries()
    discovered_ollama = [m.id for m in entries if m.backend == "ollama"]
    log.info("Discovered Ollama models (%d): %s", len(discovered_ollama), discovered_ollama)
    log.info("OpenAI-visible models (%d): %s", len(entries), [m.id for m in entries])

    if config.discover_ollama and not discovered_ollama:
        ollama_status = inspect_local_ollama_store()
        if ollama_status.warning:
            log.warning(ollama_status.warning)


def _log_torch_cuda_status() -> None:
    if not str(config.device).startswith("cuda"):
        return

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                log.warning("Configured device is %s, but torch.cuda.is_available() is false.", config.device)
                return

            device_index = _cuda_device_index(config.device)
            gpu_name = torch.cuda.get_device_name(device_index)
            props = torch.cuda.get_device_properties(device_index)
            capability = torch.cuda.get_device_capability(device_index)
            arch_list = set(getattr(torch.cuda, "get_arch_list", lambda: [])())

        vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024**3)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        supported = required_arch in arch_list or f"compute_{capability[0]}{capability[1]}" in arch_list

        if supported:
            log.info(
                "PyTorch CUDA GPU: %s (%.1f GB, compute capability %s.%s, supported by this torch build)",
                gpu_name,
                vram,
                capability[0],
                capability[1],
            )
        else:
            log.warning(
                "PyTorch sees GPU %s (%.1f GB, compute capability %s.%s), but this torch build does not include %s kernels.",
                gpu_name,
                vram,
                capability[0],
                capability[1],
                required_arch,
            )
            log.warning(
                "BetterAirLLM CUDA inference may fail. Install a PyTorch build with CUDA 12.8/13.x Blackwell support, "
                "or set AIRLLM_DEVICE=cpu for BetterAirLLM models. Ollama-proxied models are unaffected."
            )
            if arch_list:
                log.warning("This torch build reports supported CUDA archs: %s", sorted(arch_list))

        for warning in caught:
            warning_text = str(warning.message).splitlines()[0]
            if warning_text:
                log.debug("Suppressed torch CUDA warning: %s", warning_text)
    except Exception as e:
        log.warning(f"Could not query PyTorch CUDA compatibility: {e}")


def _cuda_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return 0


app = FastAPI(
    title="BetterAirLLM Server",
    description="OpenAI-compatible API for BetterAirLLM layer-wise inference",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "airllm-server"}


@app.get("/health")
async def health():
    ollama_status = inspect_local_ollama_store()
    if config.discover_ollama:
        await available_model_entries()
    cache_status = get_ollama_discovery_cache_status(config.ollama_base_url)
    version = await get_ollama_version(
        config.ollama_base_url,
        timeout_seconds=config.ollama_timeout_seconds,
    )
    running_models = await list_running_ollama_models(
        config.ollama_base_url,
        timeout_seconds=config.ollama_timeout_seconds,
    ) if version is not None else []
    return {
        "status": "ok",
        "loaded_model": loaded_model_id,
        "runtime": build_runtime_payload(include_ollama=False),
        "ollama": {
            "base_url": config.ollama_base_url,
            "discovery_enabled": config.discover_ollama,
            "daemon_available": version is not None,
            "version": version,
            "discovered_model_count": cache_status.discovered_model_count,
            "running_model_count": len(running_models),
            "running_models": _summarize_running_ollama_models(running_models),
            "last_discovery_error": cache_status.last_discovery_error,
            "last_refresh_used_stale": cache_status.last_refresh_used_stale,
            "cache_age_seconds": cache_status.cache_age_seconds,
            "models_path": ollama_status.models_path,
            "models_path_exists": ollama_status.models_path_exists,
            "blobs_found": ollama_status.blobs_found,
            "warning": ollama_status.warning,
        },
    }


def build_runtime_payload(*, include_ollama: bool = True) -> dict[str, Any]:
    with runtime_lock:
        generation = dict(last_generation_stats)
    payload = {
        "loaded_model": loaded_model_id,
        "model_manager": _get_model_manager().runtime_payload(),
        "hardware": _hardware_profile(),
        "config": {
            "stream_mode": config.stream_mode,
            "runtime_metrics_enabled": config.enable_runtime_metrics,
            "max_loaded_models": config.max_loaded_models,
            "mxfp4_execution_configured": config.mxfp4_execution,
            "mxfp4_execution_effective": _resolved_mxfp4_execution(),
            "hf_triton_module_cache_mb": config.hf_triton_module_cache_mb,
            "expert_execution_mode": config.expert_execution_mode,
            "vram_cache_mode": config.vram_cache_mode,
            "vram_cache_mb": config.vram_cache_mb,
            "cpu_cache_mb": config.cpu_cache_mb,
        },
        "hf_triton": _hf_triton_status(),
        "last_generation": generation,
    }
    if include_ollama:
        cache_status = get_ollama_discovery_cache_status(config.ollama_base_url)
        local_status = inspect_local_ollama_store()
        payload["ollama"] = {
            "base_url": config.ollama_base_url,
            "discovery_enabled": config.discover_ollama,
            "discovered_model_count": cache_status.discovered_model_count,
            "last_discovery_error": cache_status.last_discovery_error,
            "last_refresh_used_stale": cache_status.last_refresh_used_stale,
            "models_path": local_status.models_path,
            "models_path_exists": local_status.models_path_exists,
            "blobs_found": local_status.blobs_found,
            "warning": local_status.warning,
        }
    return payload


# ── GET /v1/models ───────────────────────────────────────────

@app.get("/v1/capabilities")
async def capabilities():
    """Return BetterAirLLM support and runtime capability metadata."""

    payload = build_capabilities()
    cache_status = get_ollama_discovery_cache_status(config.ollama_base_url)
    payload["ollama"]["discovered_model_count"] = cache_status.discovered_model_count
    payload["ollama"]["last_discovery_error"] = cache_status.last_discovery_error
    payload["ollama"]["last_refresh_used_stale"] = cache_status.last_refresh_used_stale
    return payload


@app.get("/v1/runtime")
async def runtime_status():
    """Return loaded model, hardware, stream, and last-generation runtime metrics."""

    return build_runtime_payload()


@app.get("/v1/models")
async def list_models():
    """Return available models in OpenAI format."""
    models = []
    for m in await available_model_entries():
        models.append({
            "id": m.id,
            "object": "model",
            "created": 0,
            "owned_by": m.owned_by,
            "permission": [],
            "root": m.id,
            "parent": None,
            "source": m.source,
            "backend": m.backend,
            "format": m.format,
            "description": m.description,
            "support": _model_support_payload(m),
            "metadata": m.metadata,
        })
    return {"object": "list", "data": models}


# ── POST /v1/chat/completions ────────────────────────────────

@app.get("/v1/models/{model_id:path}/preflight")
async def model_preflight(model_id: str):
    """Return a readiness report without loading model weights."""

    entry = await find_model_entry(model_id)
    if entry is None:
        return JSONResponse(status_code=404, content=await build_missing_model_preflight(model_id))
    return build_model_preflight(entry, requested_model_id=model_id)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests (streaming and non-streaming)."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_id = request.model
    created = int(time.time())

    entry = await find_model_entry(model_id)
    if entry is None:
        if model_id.startswith("ollama/"):
            raise HTTPException(status_code=404, detail=await _ollama_model_not_found_detail(model_id))
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found in registry. Available: {[m.id for m in await available_model_entries()]}",
        )

    if entry.backend == "ollama":
        return await handle_ollama_chat_completion(entry, request)

    try:
        model, entry = await asyncio.get_event_loop().run_in_executor(
            None, get_or_load_model, model_id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except MemoryError as e:
        raise HTTPException(status_code=507, detail=_error_detail("vram_or_ram_exhausted", str(e), model_id=model_id))
    except RuntimeError as e:
        raise _runtime_load_http_exception(e, model_id)
    except Exception as e:
        log.error(f"Model load error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_error_detail("model_load_failed", str(e), model_id=model_id))

    try:
        guard_image_request(model, entry, request.messages)
        prompt = build_prompt(model, request.messages, entry)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if request.stream:
        return StreamingResponse(
            stream_response(model, entry, prompt, request, request_id, created),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response
        try:
            output = await asyncio.get_event_loop().run_in_executor(
                None, run_inference, model, entry, prompt, request.max_tokens
            )
        except Exception as e:
            log.error(f"Inference error: {e}", exc_info=True)
            raise _inference_http_exception(e)

        with runtime_lock:
            usage_stats = dict(last_generation_stats)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": entry.id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage_stats.get("prompt_tokens", -1),
                "completion_tokens": usage_stats.get("completion_tokens", -1),
                "total_tokens": usage_stats.get("total_tokens", -1),
            },
        }


async def handle_ollama_chat_completion(entry: ModelEntry, request: ChatCompletionRequest):
    """Proxy an OpenAI-compatible chat request to the local Ollama daemon."""

    payload = request.model_dump(by_alias=True, exclude_none=True)
    payload["model"] = entry.ollama_model or raw_ollama_model_name(request.model)

    if request.stream:
        return StreamingResponse(
            ollama_stream_response(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        return await proxy_ollama_chat_completion(
            config.ollama_base_url,
            payload,
            timeout_seconds=max(30.0, config.ollama_timeout_seconds),
        )
    except OllamaHTTPError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=f"Ollama backend unavailable: {e}")


async def ollama_stream_response(payload: dict[str, Any]):
    try:
        async for chunk in stream_ollama_chat_completion(
            config.ollama_base_url,
            payload,
            timeout_seconds=max(30.0, config.ollama_timeout_seconds),
        ):
            yield chunk
    except Exception as e:  # noqa: BLE001 - convert upstream stream failures to SSE.
        import json as json_mod

        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": payload.get("model", ""),
            "choices": [{"index": 0, "delta": {"content": f"[Ollama error: {e}]"}, "finish_reason": "stop"}],
        }
        yield f"data: {json_mod.dumps(error_chunk)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


async def _ollama_model_not_found_detail(model_id: str) -> str:
    raw_name = raw_ollama_model_name(model_id)
    entries = await available_model_entries()
    available = sorted(e.ollama_model or raw_ollama_model_name(e.id) for e in entries if e.backend == "ollama")
    local_status = inspect_local_ollama_store()
    cache_status = get_ollama_discovery_cache_status(config.ollama_base_url)
    if local_status.blobs_found and not available:
        return (
            f"Ollama blobs exist locally, but the Ollama daemon is unavailable at {config.ollama_base_url}. "
            "Start Ollama and retry model discovery."
        )
    suffix = f" Available Ollama models: {available}" if available else " No Ollama models are currently discoverable."
    if cache_status.last_discovery_error:
        suffix += f" Last discovery error: {cache_status.last_discovery_error}"
    return f"Ollama model '{raw_name}' is not installed or not discoverable.{suffix}"


def _summarize_running_ollama_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized = []
    for item in models:
        summarized.append({
            "name": item.get("name") or item.get("model"),
            "model": item.get("model"),
            "size": item.get("size"),
            "size_vram": item.get("size_vram"),
            "context_length": item.get("context_length"),
            "expires_at": item.get("expires_at"),
            "details": item.get("details"),
        })
    return summarized


async def stream_response(model, entry, prompt, request, request_id, created):
    """Generate SSE stream in OpenAI format."""

    if str(config.stream_mode).lower() != "compat":
        yielded_any = False
        try:
            event_queue = run_inference_stream(model, entry, prompt, request.max_tokens)
            while True:
                kind, payload = await asyncio.to_thread(event_queue.get)
                if kind == "token":
                    yielded_any = True
                    yield _sse_chat_chunk(request_id, created, entry.id, {"content": payload})
                elif kind == "done":
                    yield _sse_final_chunk(request_id, created, entry.id)
                    yield "data: [DONE]\n\n"
                    return
                elif kind == "error":
                    if not yielded_any and str(config.stream_mode).lower() == "auto":
                        log.warning("Real stream failed before first token; falling back to compat streaming: %s", payload)
                        async for chunk in _compat_stream_response(model, entry, prompt, request, request_id, created):
                            yield chunk
                        return
                    raise payload
        except Exception as e:
            log.error("Stream inference error: %s", e, exc_info=True)
            yield _sse_chat_chunk(request_id, created, entry.id, {"content": f"[Error: {e}]"}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

    async for chunk in _compat_stream_response(model, entry, prompt, request, request_id, created):
        yield chunk


async def _compat_stream_response(model, entry, prompt, request, request_id, created):
    """Compatibility stream that chunks a completed generation."""

    try:
        output = await asyncio.get_event_loop().run_in_executor(
            None, run_inference, model, entry, prompt, request.max_tokens
        )
    except Exception as e:
        log.error(f"Stream inference error: {e}", exc_info=True)
        yield _sse_chat_chunk(request_id, created, entry.id, {"content": f"[Error: {e}]"}, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    words = output.split(" ")
    for i, word in enumerate(words):
        token = word if i == 0 else " " + word
        yield _sse_chat_chunk(request_id, created, entry.id, {"content": token})
        await asyncio.sleep(0.02)

    yield _sse_final_chunk(request_id, created, entry.id)
    yield "data: [DONE]\n\n"


def _sse_chat_chunk(request_id: str, created: int, model_id: str, delta: dict[str, Any], finish_reason: Optional[str] = None) -> str:
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _sse_final_chunk(request_id: str, created: int, model_id: str) -> str:
    return _sse_chat_chunk(request_id, created, model_id, {}, finish_reason="stop")


# ═════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _config = load_config()
    _airllm_count = len([m for m in _config.models if m.backend == "airllm"])
    _configured_ollama_count = len([m for m in _config.models if m.backend == "ollama"])
    print(f"""
=================================================
  BetterAirLLM Server v1.0.0
  OpenAI-compatible API for layer-wise inference
=================================================
  Endpoint:  http://localhost:{_config.port}/v1
  BetterAirLLM config models:     {_airllm_count}
  Ollama config models:     {_configured_ollama_count}
  Ollama discovery:         {"enabled" if _config.discover_ollama else "disabled"}
  Full model inventory:     shown in startup logs below
  Configured device:        {_config.device}
=================================================

Connect Open WebUI:
  Settings > Connections > Add OpenAI connection
  URL: http://localhost:{_config.port}/v1
  Key: sk-airllm (any value works)
""")
    uvicorn.run(
        "server:app",
        host=_config.host,
        port=_config.port,
        log_level="info",
    )
