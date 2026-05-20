# betterairllm/architecture_prober.py
"""ArchitectureProber
~~~~~~~~~~~~~~~~~~~~~~
Detects the internal layout of a HuggingFace transformer model in an
architecture‑agnostic way.

The class inspects a model's ``AutoConfig`` and its ``nn.Module`` instance
to discover the attribute that holds the sequential layers, the embedding
layer and the language‑model head.  It falls back to a hard‑coded mapping
for the most common model families (Llama, Qwen2, Mistral, Gemma, Phi).

The detection logic follows the "Dynamic Detection" directive:
- Prefer explicit config information when available.
- Otherwise, look for the *longest* ``nn.ModuleList`` (or ``list`` of
  ``nn.Module``) attribute.
- If that fails, scan ``named_modules()`` for the most frequently
  occurring module type (the typical "block" pattern) and use a fallback
  mapping.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch.nn as nn
from transformers import PretrainedConfig

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback mapping for well‑known architectures
# ---------------------------------------------------------------------------
_FALLBACK_MAPPING: Dict[str, Dict[str, str]] = {
    "llama": {
        "layer_list": "model.layers",
        "embedding": "model.embed_tokens",
        "head": "model.lm_head",
    },
    "qwen2": {
        "layer_list": "model.layers",
        "embedding": "model.embed_tokens",
        "head": "model.lm_head",
    },
    "mistral": {
        "layer_list": "model.layers",
        "embedding": "model.embed_tokens",
        "head": "model.lm_head",
    },
    "gemma": {
        "layer_list": "model.layers",
        "embedding": "model.embed_tokens",
        "head": "model.lm_head",
    },
    "phi": {
        "layer_list": "model.layers",
        "embedding": "model.wte",
        "head": "model.lm_head",
    },
}


class BetterAirLLMError(RuntimeError):
    """Custom exception for BetterAirLLM‑related failures."""


class ArchitectureProber:
    """Detects model architecture details.

    Parameters
    ----------
    config:
        The ``PretrainedConfig`` associated with the model.
    model:
        The instantiated ``nn.Module``. Required for runtime inspection;
        may be ``None`` if only static config information is needed.
    """

    def __init__(self, config: PretrainedConfig, model: Optional[nn.Module] = None) -> None:
        self.config = config
        self.model = model
        self.model_type: str = getattr(config, "model_type", "unknown").lower()
        self._layer_list_name: Optional[str] = None
        self._embedding_name: Optional[str] = None
        self._head_name: Optional[str] = None
        self._probe()

    # ---------------------------------------------------------------------
    # Public getters
    # ---------------------------------------------------------------------
    @property
    def layer_list_name(self) -> str:
        if self._layer_list_name is None:
            raise BetterAirLLMError("Layer list name could not be resolved.")
        return self._layer_list_name

    @property
    def embedding_name(self) -> str:
        if self._embedding_name is None:
            raise BetterAirLLMError("Embedding layer name could not be resolved.")
        return self._embedding_name

    @property
    def head_name(self) -> str:
        if self._head_name is None:
            raise BetterAirLLMError("Head layer name could not be resolved.")
        return self._head_name

    # ---------------------------------------------------------------------
    # Internal probing helpers
    # ---------------------------------------------------------------------
    def _probe(self) -> None:
        """Run the detection workflow.

        The order is:
        1. Try explicit config‑driven mapping.
        2. Inspect the model object for a ``nn.ModuleList``.
        3. Fall back to the hard‑coded table.
        """
        # 1. Config‑driven mapping – some configs expose the attr name.
        explicit = getattr(self.config, "layers_attribute", None)
        if explicit:
            self._apply_mapping(explicit)
            return

        # 2. Runtime inspection (requires a model instance).
        if self.model is not None:
            self._detect_via_model()
            if self._layer_list_name:
                return

        # 3. Fallback table based on model_type.
        self._apply_fallback()

    def _apply_mapping(self, layer_attr: str) -> None:
        """Apply a user‑provided ``layer_attr`` if it exists on the model."""
        if not hasattr(self.model, layer_attr):
            _logger.debug("Explicit layer attribute %s not found on model.", layer_attr)
            return
        self._layer_list_name = layer_attr
        # Guess embedding / head using common suffixes.
        self._embedding_name = getattr(self.config, "embeddings_attribute", "model.embed_tokens")
        self._head_name = getattr(self.config, "head_attribute", "model.lm_head")

    def _detect_via_model(self) -> None:
        """Detect the sequential layer container by scanning model attributes."""
        candidates = []
        for name, attr in vars(self.model).items():
            if isinstance(attr, (nn.ModuleList, list)) and len(attr) > 1:
                # Ensure the container holds ``nn.Module`` instances.
                if all(isinstance(m, nn.Module) for m in attr):
                    candidates.append((name, len(attr)))
        if candidates:
            # Choose the longest list – typical for transformer blocks.
            candidates.sort(key=lambda x: x[1], reverse=True)
            self._layer_list_name = candidates[0][0]
            # Try to locate embedding and head via common attribute names.
            self._embedding_name = self._find_embedding()
            self._head_name = self._find_head()
            return

        # Heuristic fallback using named_modules – find the most common block type.
        type_counts: Dict[type, list] = {}
        for name, module in self.model.named_modules():
            t = type(module)
            type_counts.setdefault(t, []).append(name)
        # Exclude the top‑level model class.
        if len(type_counts) > 1:
            most_common = max(type_counts.items(), key=lambda kv: len(kv[1]))
            # If the most common type appears more than once we assume it's the block.
            if len(most_common[1]) > 1:
                # Derive a plausible attribute name from the first occurrence.
                exemplar = most_common[1][0]
                prefix = exemplar.split(".")[0]
                self._layer_list_name = prefix
                self._embedding_name = self._find_embedding()
                self._head_name = self._find_head()

    def _find_embedding(self) -> str:
        # Common names – try a small ordered list.
        for cand in ("embed_tokens", "wte", "embeddings", "word_embeddings"):
            if hasattr(self.model, cand):
                return f"model.{cand}"
        # Fallback to generic.
        return "model.embed_tokens"

    def _find_head(self) -> str:
        for cand in ("lm_head", "head", "output_head"):
            if hasattr(self.model, cand):
                return f"model.{cand}"
        return "model.lm_head"

    def _apply_fallback(self) -> None:
        mapping = _FALLBACK_MAPPING.get(self.model_type)
        if mapping:
            self._layer_list_name = mapping["layer_list"]
            self._embedding_name = mapping["embedding"]
            self._head_name = mapping["head"]
        else:
            msg = f"Unsupported model_type '{self.model_type}' and no fallback available."
            _logger.error(msg)
            raise BetterAirLLMError(msg)

    # ---------------------------------------------------------------------
    # Convenience string representation
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ArchitectureProber(model_type={self.model_type}, "
            f"layer_list_name={self._layer_list_name}, "
            f"embedding_name={self._embedding_name}, "
            f"head_name={self._head_name})"
        )
