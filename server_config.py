"""BetterAirLLM OpenAI-Compatible API Server - Configuration.

Multi-model registry with environment variable overrides.
Models can be added via AIRLLM_MODELS env var (JSON) or the default registry.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelEntry:
    """A single model available for inference."""

    id: str
    repo_id: str
    description: str = ""
    compression: Optional[str] = None
    max_seq_len: int = 2048
    owned_by: str = "airllm"
    source: str = "hf"
    backend: str = "airllm"
    format: Optional[str] = None
    ollama_model: Optional[str] = None
    family: Optional[str] = None
    prompt_format: Optional[str] = None
    status: str = "configured"
    tested_level: Optional[str] = None
    quantization: Optional[str] = None
    memory: dict[str, Any] = field(default_factory=dict)
    context_length: Optional[int] = None
    known_limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerConfig:
    """Server-wide configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda:0"
    dense_device: Optional[str] = None
    dtype: str = "float16"
    hf_token: Optional[str] = None
    prefetching: bool = True
    use_harmony_prompt: bool = True
    reasoning_effort: str = "low"
    max_vram_mb: int = 7600
    mxfp4_device: str = "cuda"
    expert_matmul_device: str = "cuda"
    vram_cache_mode: str = "packed_experts"
    vram_cache_mb: int = 4000
    cpu_cache_mb: int = 12000
    layer_cleanup_interval: int = 8
    mxfp4_execution: str = "reference_cuda"
    hf_triton_module_cache_mb: int = 0
    expert_execution_mode: str = "grouped_by_expert"
    os_reserved_ram_mb: int = 8192
    quiet_progress: bool = True
    max_loaded_models: int = 1
    stream_mode: str = "auto"
    enable_runtime_metrics: bool = True
    ollama_base_url: str = "http://localhost:11434"
    discover_ollama: bool = True
    ollama_timeout_seconds: float = 5.0
    ollama_discovery_ttl_seconds: float = 10.0
    models: list[ModelEntry] = field(default_factory=list)


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_mxfp4_execution() -> str:
    """Prefer the HF Triton path only when CUDA is visible at config time."""

    try:
        import torch

        return "hf_triton" if torch.cuda.is_available() else "reference_cuda"
    except Exception:
        return "reference_cuda"


def _normalized_stream_mode(value: str) -> str:
    mode = (value or "auto").strip().lower()
    if mode not in {"auto", "true", "compat"}:
        print(f"[WARN] Unsupported AIRLLM_STREAM_MODE={value!r}; using auto")
        return "auto"
    return mode


def load_default_models(registry_path: Optional[str | Path] = None) -> list[ModelEntry]:
    """Load the default model registry from JSON."""

    path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY_PATH
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise TypeError("model registry must be a JSON array")
    return [_model_entry_from_dict(item) for item in raw]


def _model_entry_from_dict(item: dict[str, Any]) -> ModelEntry:
    """Load old and new registry entries without breaking minimal JSON."""

    item = dict(item)
    metadata = dict(item.get("metadata") or {})
    for key in (
        "family",
        "prompt_format",
        "status",
        "tested_level",
        "quantization",
        "memory",
        "context_length",
        "known_limitations",
    ):
        if key in item and item[key] is not None:
            metadata.setdefault(key, item[key])
    item["metadata"] = metadata
    return ModelEntry(**item)


def load_config() -> ServerConfig:
    """Build config from environment variables + defaults."""

    config = ServerConfig(
        host=os.getenv("AIRLLM_HOST", "0.0.0.0"),
        port=int(os.getenv("AIRLLM_PORT", "8000")),
        device=os.getenv("AIRLLM_DEVICE", "cuda:0"),
        dense_device=os.getenv("AIRLLM_DENSE_DEVICE"),
        dtype=os.getenv("AIRLLM_DTYPE", "float16"),
        hf_token=os.getenv("HF_TOKEN"),
        prefetching=_bool_env("AIRLLM_PREFETCH", True),
        use_harmony_prompt=_bool_env("AIRLLM_USE_HARMONY_PROMPT", True),
        reasoning_effort=os.getenv("AIRLLM_REASONING_EFFORT", "low"),
        max_vram_mb=int(os.getenv("AIRLLM_MAX_VRAM_MB", "7600")),
        mxfp4_device=os.getenv("AIRLLM_MXFP4_DEVICE", "cuda"),
        expert_matmul_device=os.getenv("AIRLLM_EXPERT_MATMUL_DEVICE", "cuda"),
        vram_cache_mode=os.getenv("AIRLLM_VRAM_CACHE_MODE", "packed_experts"),
        vram_cache_mb=int(os.getenv("AIRLLM_VRAM_CACHE_MB", "4000")),
        cpu_cache_mb=int(os.getenv("AIRLLM_CPU_CACHE_MB", "12000")),
        layer_cleanup_interval=int(os.getenv("AIRLLM_LAYER_CLEANUP_INTERVAL", "8")),
        mxfp4_execution=os.getenv("AIRLLM_MXFP4_EXECUTION", _default_mxfp4_execution()),
        hf_triton_module_cache_mb=int(os.getenv("AIRLLM_HF_TRITON_MODULE_CACHE_MB", "0")),
        expert_execution_mode=os.getenv("AIRLLM_EXPERT_EXECUTION_MODE", "grouped_by_expert"),
        os_reserved_ram_mb=int(os.getenv("AIRLLM_OS_RESERVED_RAM_MB", "8192")),
        quiet_progress=_bool_env("AIRLLM_QUIET_PROGRESS", True),
        max_loaded_models=max(1, int(os.getenv("AIRLLM_MAX_LOADED_MODELS", "1"))),
        stream_mode=_normalized_stream_mode(os.getenv("AIRLLM_STREAM_MODE", "auto")),
        enable_runtime_metrics=_bool_env("AIRLLM_ENABLE_RUNTIME_METRICS", True),
        ollama_base_url=os.getenv("AIRLLM_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        discover_ollama=_bool_env("AIRLLM_DISCOVER_OLLAMA", True),
        ollama_timeout_seconds=float(os.getenv("AIRLLM_OLLAMA_TIMEOUT_SECONDS", "5.0")),
        ollama_discovery_ttl_seconds=float(os.getenv("AIRLLM_OLLAMA_DISCOVERY_TTL_SECONDS", "10.0")),
    )

    models_json = os.getenv("AIRLLM_MODELS")
    if models_json:
        try:
            raw = json.loads(models_json)
            config.models = [_model_entry_from_dict(m) for m in raw]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[WARN] Failed to parse AIRLLM_MODELS: {e}, using defaults")
            config.models = load_default_models()
    else:
        config.models = load_default_models()

    extra_model = os.getenv("AIRLLM_MODEL")
    if extra_model:
        extra_id = os.getenv("AIRLLM_MODEL_ID", extra_model.split("/")[-1].lower())
        extra_compression = os.getenv("AIRLLM_COMPRESSION")
        extra_max_seq = int(os.getenv("AIRLLM_MAX_SEQ_LEN", "2048"))
        entry = ModelEntry(
            id=extra_id,
            repo_id=extra_model,
            description=f"Custom model: {extra_model}",
            compression=extra_compression if extra_compression != "None" else None,
            max_seq_len=extra_max_seq,
        )
        config.models.insert(0, entry)

    return config
