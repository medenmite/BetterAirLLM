"""Benchmark and resume BetterAirLLM incremental MoE split construction."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AIR_LLM_ROOT = REPO_ROOT / "air_llm"
if str(AIR_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(AIR_LLM_ROOT))

from betterairllm.moe_layout_probe import load_config  # noqa: E402
from betterairllm.moe_split_builder import IncrementalMoESplitBuilder  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402


def _checkpoint_path(model, cache_dir=None, token=None):
    if Path(model).exists():
        return Path(model)
    config_path = Path(hf_hub_download(model, "config.json", cache_dir=cache_dir, token=token))
    return config_path.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--hf-token")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--layer-name", action="append", help="Build or verify a specific split layer, e.g. model.layers.0.")
    parser.add_argument("--resume-split-build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--moe-strict-streaming", action="store_true", default=True)
    parser.add_argument("--adapter-name", default=None)
    args = parser.parse_args()

    checkpoint = _checkpoint_path(args.model, cache_dir=args.hf_cache_dir, token=args.hf_token)
    config = load_config(str(checkpoint), cache_dir=args.hf_cache_dir, token=args.hf_token)
    adapter_name = args.adapter_name
    if adapter_name is None and config.get("model_type") == "gpt_oss":
        adapter_name = "gpt_oss_mxfp4_reference"

    split_dir = Path(args.output_dir) if args.output_dir else checkpoint / "splitted_model.moe"
    if args.force and split_dir.exists():
        shutil.rmtree(split_dir)

    started = time.perf_counter()
    builder = IncrementalMoESplitBuilder(
        checkpoint,
        split_dir,
        repo_id=args.model if not Path(args.model).exists() else None,
        hf_token=args.hf_token,
        strict_mode=args.moe_strict_streaming,
        adapter_name=adapter_name,
    )
    before = builder.status()
    if args.layer_name:
        for layer_name in args.layer_name:
            builder.ensure_layer(layer_name)
            builder.verify_layer(layer_name)
    else:
        builder.build_until_ready(max_layers=args.max_layers)
    elapsed = time.perf_counter() - started
    after = builder.status()
    reused_layers = max(0, before["completed_layers"])
    total_layers = after["total_layers"]
    disk_usage = sum(path.stat().st_size for path in split_dir.rglob("*") if path.is_file())
    report = {
        "model": args.model,
        "split_dir": str(split_dir),
        "total_split_time_seconds": elapsed,
        "bytes_processed_per_second": after["bytes_per_second"],
        "layers_per_minute": (after["completed_layers"] - before["completed_layers"]) / elapsed * 60 if elapsed > 0 else 0,
        "shard_reuse_hit_rate": reused_layers / total_layers if total_layers else 0,
        "estimated_remaining_time_seconds": after["estimated_remaining_seconds"],
        "packed_mxfp4_bytes_processed": after["bytes_processed"],
        "dense_bytes_avoided": None,
        "disk_usage_bytes": disk_usage,
        "before": before,
        "after": after,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
