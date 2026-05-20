"""Ollama discovery and proxy helpers for the BetterAirLLM server."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from server_config import ModelEntry


OLLAMA_ID_PREFIX = "ollama/"
_DISCOVERY_CACHE: dict[str, "OllamaDiscoveryCacheEntry"] = {}


@dataclass
class OllamaLocalStatus:
    daemon_available: bool
    models_path: str
    models_path_exists: bool
    blobs_found: bool
    warning: Optional[str] = None


@dataclass
class OllamaDiscoveryCacheEntry:
    entries: list[ModelEntry]
    refreshed_at: float
    last_error: Optional[str] = None
    last_refresh_used_stale: bool = False


@dataclass
class OllamaDiscoveryCacheStatus:
    has_cache: bool
    cache_age_seconds: Optional[float]
    discovered_model_count: int
    last_discovery_error: Optional[str]
    last_refresh_used_stale: bool


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama daemon cannot be reached."""


class OllamaHTTPError(RuntimeError):
    """Raised when Ollama returns a non-2xx HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def ollama_entry_id(model_name: str) -> str:
    return f"{OLLAMA_ID_PREFIX}{model_name}"


def raw_ollama_model_name(model_id: str) -> str:
    if model_id.startswith(OLLAMA_ID_PREFIX):
        return model_id[len(OLLAMA_ID_PREFIX):]
    return model_id


def _default_models_path() -> Path:
    configured = os.getenv("OLLAMA_MODELS")
    if configured:
        return Path(configured)
    return Path.home() / ".ollama" / "models"


def inspect_local_ollama_store(models_path: Optional[Path] = None) -> OllamaLocalStatus:
    path = models_path or _default_models_path()
    blobs_path = path / "blobs"
    blobs_found = False
    if blobs_path.exists():
        try:
            blobs_found = any(blobs_path.iterdir())
        except OSError:
            blobs_found = False
    return OllamaLocalStatus(
        daemon_available=False,
        models_path=str(path),
        models_path_exists=path.exists(),
        blobs_found=blobs_found,
        warning=(
            "Ollama model blobs exist locally, but the Ollama daemon is unavailable."
            if blobs_found else None
        ),
    )


async def discover_ollama_models(
    base_url: str,
    *,
    timeout_seconds: float = 5.0,
    ttl_seconds: float = 0.0,
    force_refresh: bool = False,
) -> list[ModelEntry]:
    """Discover installed Ollama models through the local daemon."""

    base_url = base_url.rstrip("/")
    now = time.monotonic()
    cached = _DISCOVERY_CACHE.get(base_url)
    if (
        cached is not None
        and ttl_seconds > 0
        and not force_refresh
        and now - cached.refreshed_at <= ttl_seconds
    ):
        return _copy_entries(cached.entries)

    try:
        discovered = await _discover_ollama_models_uncached(base_url, timeout_seconds)
        _DISCOVERY_CACHE[base_url] = OllamaDiscoveryCacheEntry(
            entries=_copy_entries(discovered),
            refreshed_at=time.monotonic(),
        )
        return discovered
    except (OllamaHTTPError, OSError, json.JSONDecodeError) as exc:
        if cached is not None:
            cached.last_error = str(exc)
            cached.last_refresh_used_stale = True
            return _copy_entries(cached.entries)
        raise OllamaUnavailable(str(exc)) from exc


async def _discover_ollama_models_uncached(base_url: str, timeout_seconds: float) -> list[ModelEntry]:
    tags_payload = await asyncio.to_thread(
        _request_json,
        "GET",
        f"{base_url}/api/tags",
        None,
        timeout_seconds,
    )
    discovered = []
    for item in tags_payload.get("models", []):
        name = item.get("model") or item.get("name")
        if not name:
            continue
        details = item.get("details") or {}
        show_payload = await _show_model_details(base_url, name, timeout_seconds)
        merged_details = {**details, **(show_payload.get("details") or {})}
        model_info = show_payload.get("model_info") or {}
        context_length = _find_context_length(model_info)
        metadata = _build_ollama_metadata(item, show_payload, merged_details, model_info, context_length)
        discovered.append(
            ModelEntry(
                id=ollama_entry_id(name),
                repo_id=name,
                description=_describe_ollama_model(name, merged_details),
                compression=None,
                max_seq_len=context_length or 2048,
                owned_by="ollama",
                source="ollama",
                backend="ollama",
                format=merged_details.get("format") or "unknown",
                ollama_model=name,
                metadata=metadata,
            )
        )
    return discovered


async def _show_model_details(base_url: str, name: str, timeout_seconds: float) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(
            _request_json,
            "POST",
            f"{base_url}/api/show",
            {"model": name},
            timeout_seconds,
        )
    except (OllamaHTTPError, OSError, json.JSONDecodeError):
        return {}


async def get_ollama_version(base_url: str, *, timeout_seconds: float = 5.0) -> Optional[str]:
    try:
        payload = await asyncio.to_thread(
            _request_json,
            "GET",
            f"{base_url.rstrip('/')}/api/version",
            None,
            timeout_seconds,
        )
        version = payload.get("version")
        return str(version) if version is not None else None
    except (OllamaHTTPError, OSError, json.JSONDecodeError):
        return None


async def list_running_ollama_models(base_url: str, *, timeout_seconds: float = 5.0) -> list[dict[str, Any]]:
    try:
        payload = await asyncio.to_thread(
            _request_json,
            "GET",
            f"{base_url.rstrip('/')}/api/ps",
            None,
            timeout_seconds,
        )
        models = payload.get("models", [])
        return models if isinstance(models, list) else []
    except (OllamaHTTPError, OSError, json.JSONDecodeError):
        return []


def get_ollama_discovery_cache_status(base_url: str) -> OllamaDiscoveryCacheStatus:
    cache = _DISCOVERY_CACHE.get(base_url.rstrip("/"))
    if cache is None:
        return OllamaDiscoveryCacheStatus(
            has_cache=False,
            cache_age_seconds=None,
            discovered_model_count=0,
            last_discovery_error=None,
            last_refresh_used_stale=False,
        )
    return OllamaDiscoveryCacheStatus(
        has_cache=True,
        cache_age_seconds=max(0.0, time.monotonic() - cache.refreshed_at),
        discovered_model_count=len(cache.entries),
        last_discovery_error=cache.last_error,
        last_refresh_used_stale=cache.last_refresh_used_stale,
    )


def clear_ollama_discovery_cache() -> None:
    _DISCOVERY_CACHE.clear()


def _request_json(method: str, url: str, payload: Optional[dict[str, Any]], timeout_seconds: float) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urlrequest.Request(url, data=body, headers=headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urlerror.HTTPError as exc:
        detail = _read_error_detail(exc)
        raise OllamaHTTPError(exc.code, detail) from exc
    except urlerror.URLError as exc:
        raise OSError(str(exc.reason)) from exc


def _read_error_detail(exc: urlerror.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8")
        payload = json.loads(body)
        if isinstance(payload, dict):
            return str(payload.get("error") or payload.get("detail") or payload)
        return body
    except Exception:  # noqa: BLE001
        return str(exc)


def _describe_ollama_model(name: str, details: dict[str, Any]) -> str:
    parts = ["Ollama model", name]
    family = details.get("family")
    parameter_size = details.get("parameter_size")
    quantization = details.get("quantization_level")
    if family:
        parts.append(family)
    if parameter_size:
        parts.append(str(parameter_size))
    if quantization:
        parts.append(str(quantization))
    return " - ".join(parts)


def _build_ollama_metadata(
    tag_item: dict[str, Any],
    show_payload: dict[str, Any],
    details: dict[str, Any],
    model_info: dict[str, Any],
    context_length: Optional[int],
) -> dict[str, Any]:
    metadata = {
        "format": details.get("format"),
        "family": details.get("family"),
        "families": details.get("families"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "size": tag_item.get("size"),
        "digest": tag_item.get("digest"),
        "modified_at": tag_item.get("modified_at") or show_payload.get("modified_at"),
        "capabilities": show_payload.get("capabilities"),
        "context_length": context_length,
        "model_info": _select_model_info(model_info),
    }
    for optional_key in ("parameters", "template"):
        if show_payload.get(optional_key):
            metadata[optional_key] = show_payload[optional_key]
    return {key: value for key, value in metadata.items() if value is not None}


def _select_model_info(model_info: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    allowed_exact = {
        "general.architecture",
        "general.file_type",
        "general.parameter_count",
        "general.quantization_version",
        "tokenizer.ggml.model",
    }
    for key, value in model_info.items():
        if key in allowed_exact or key.endswith(".context_length"):
            selected[key] = value
    return selected


def _find_context_length(model_info: dict[str, Any]) -> Optional[int]:
    for key, value in model_info.items():
        if key.endswith(".context_length") or key == "context_length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _copy_entries(entries: list[ModelEntry]) -> list[ModelEntry]:
    return copy.deepcopy(entries)


async def proxy_ollama_chat_completion(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Forward a non-streaming OpenAI-compatible chat request to Ollama."""

    try:
        return await asyncio.to_thread(
            _request_json,
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            payload,
            timeout_seconds,
        )
    except OllamaHTTPError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise OllamaUnavailable(str(exc)) from exc


async def stream_ollama_chat_completion(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 300.0,
) -> AsyncIterator[bytes]:
    """Forward an OpenAI-compatible streaming chat request to Ollama."""

    req = _build_json_request(f"{base_url.rstrip('/')}/v1/chat/completions", payload)
    try:
        response = await asyncio.to_thread(lambda: urlrequest.urlopen(req, timeout=timeout_seconds))
    except urlerror.HTTPError as exc:
        detail = _read_error_detail(exc)
        raise OllamaHTTPError(exc.code, detail) from exc
    except urlerror.URLError as exc:
        raise OllamaUnavailable(str(exc.reason)) from exc
    try:
        while True:
            chunk = await asyncio.to_thread(response.read, 8192)
            if not chunk:
                break
            yield chunk
    except urlerror.HTTPError as exc:
        detail = _read_error_detail(exc)
        raise OllamaHTTPError(exc.code, detail) from exc
    except urlerror.URLError as exc:
        raise OllamaUnavailable(str(exc.reason)) from exc
    finally:
        response.close()


def _build_json_request(url: str, payload: dict[str, Any]) -> urlrequest.Request:
    return urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
