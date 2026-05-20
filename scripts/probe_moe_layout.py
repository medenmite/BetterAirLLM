"""Probe MoE checkpoint layout without downloading full model shards by default."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AIR_LLM_ROOT = REPO_ROOT / "air_llm"
if str(AIR_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(AIR_LLM_ROOT))

if "airllm" not in sys.modules:
    package = types.ModuleType("airllm")
    package.__path__ = [str(AIR_LLM_ROOT / "airllm")]
    sys.modules["airllm"] = package

_probe_spec = importlib.util.spec_from_file_location(
    "airllm.moe_layout_probe",
    AIR_LLM_ROOT / "airllm" / "moe_layout_probe.py",
)
_probe_module = importlib.util.module_from_spec(_probe_spec)
sys.modules["airllm.moe_layout_probe"] = _probe_module
_probe_spec.loader.exec_module(_probe_module)
download_gpt_oss_one_expert_mxfp4_smoke = _probe_module.download_gpt_oss_one_expert_mxfp4_smoke
probe_moe_layout = _probe_module.probe_moe_layout


def _run_one_layer_smoke(args, probe):
    if probe.get("manifest_verification", {}).get("adapter_name") == "qwen3_5_moe":
        return _run_qwen35_one_layer_smoke(args, probe)

    if "gpt-oss-20b" not in args.model.lower():
        raise SystemExit("--download-one-layer is only allowed for openai/gpt-oss-20b-style repos")

    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        import torch
    except ImportError as exc:
        raise SystemExit(
            "--download-one-layer requires huggingface_hub, safetensors, and torch. "
            "Install requirements-dev.txt first."
        ) from exc

    from airllm.moe_layout_probe import (
        GPT_OSS_DENSE_EXPERT_SUFFIXES,
        GPT_OSS_PACKED_EXPERT_SUFFIXES,
        GPT_OSS_ROUTER_SUFFIXES,
        load_safetensors_index,
    )
    from airllm.selective_fused_moe import (
        GptOssSelectiveFusedMoEAdapter,
        build_fake_gpt_oss_expert_shards,
        fake_gpt_oss_full_fused_moe_forward,
    )

    index = load_safetensors_index(args.model, cache_dir=args.cache_dir, token=args.hf_token)
    weight_map = index.get("weight_map") or {}
    sample_layers = probe.get("sample_expected_layer_tensors") or {}
    if not sample_layers:
        raise SystemExit("No complete GPT-OSS MoE layer was found in index metadata")
    layer_prefix, layer_info = next(iter(sample_layers.items()))
    layout_kind = layer_info.get("layout_kind")
    if layout_kind == "mxfp4_packed":
        expert_suffixes = GPT_OSS_PACKED_EXPERT_SUFFIXES
    elif layout_kind == "dense":
        expert_suffixes = GPT_OSS_DENSE_EXPERT_SUFFIXES
    else:
        raise SystemExit(f"Cannot smoke test {layer_prefix}; unsupported layer layout: {layout_kind}")
    tensor_names = [layer_prefix + suffix for suffix in (*GPT_OSS_ROUTER_SUFFIXES, *expert_suffixes)]
    missing = [name for name in tensor_names if name not in weight_map]
    if missing:
        raise SystemExit(f"Cannot smoke test {layer_prefix}; missing tensors: {missing}")

    shard_names = sorted({weight_map[name] for name in tensor_names})
    shard_paths = {
        shard: Path(hf_hub_download(args.model, shard, cache_dir=args.cache_dir, token=args.hf_token))
        for shard in shard_names
    }

    tensor_shapes = {}
    for name in tensor_names:
        with safe_open(str(shard_paths[weight_map[name]]), framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(name)
            tensor_shapes[name] = list(tensor_slice.get_shape())

    if layout_kind == "mxfp4_packed":
        router_tensors = {}
        for name in [layer_prefix + suffix for suffix in GPT_OSS_ROUTER_SUFFIXES]:
            with safe_open(str(shard_paths[weight_map[name]]), framework="pt", device="cpu") as handle:
                router_tensors[name] = handle.get_tensor(name)
        router_weight = router_tensors[layer_prefix + ".mlp.router.weight"]
        router_bias = router_tensors.get(layer_prefix + ".mlp.router.bias")
        hidden = torch.randn(1, min(2, args.smoke_tokens), int(probe["hidden_size"]), dtype=torch.float32)
        logits = torch.nn.functional.linear(hidden.reshape(-1, hidden.shape[-1]), router_weight.float(), None if router_bias is None else router_bias.float())
        _, router_indices = torch.topk(logits, int(probe["experts_per_token"]), dim=-1)
        selected = sorted(int(item) for item in torch.unique(router_indices).tolist())
        return {
            "one_layer_smoke": "shape_dtype_only",
            "reason": "real checkpoint uses MXFP4 packed blocks/scales; packed selective matmul is not claimed here",
            "downloaded_shards": shard_names,
            "layer_prefix": layer_prefix,
            "layout_kind": layout_kind,
            "tensor_shapes": tensor_shapes,
            "selected_experts_from_router": selected,
            "selected_expert_request_plan_only": True,
            "strict_full_fused_tensor_loaded": False,
        }

    tensors = {}
    try:
        for name in tensor_names:
            with safe_open(str(shard_paths[weight_map[name]]), framework="pt", device="cpu") as handle:
                tensors[name] = handle.get_tensor(name)
    except RuntimeError as exc:
        return {
            "one_layer_smoke": "shape_dtype_only",
            "reason": f"tensor materialization failed: {exc}",
            "downloaded_shards": shard_names,
            "layer_prefix": layer_prefix,
            "strict_full_fused_tensor_loaded": False,
        }

    router_weight = tensors[layer_prefix + ".mlp.router.weight"]
    router_bias = tensors.get(layer_prefix + ".mlp.router.bias")
    gate_up = tensors[layer_prefix + ".mlp.experts.gate_up_proj"]
    gate_up_bias = tensors[layer_prefix + ".mlp.experts.gate_up_proj_bias"]
    down = tensors[layer_prefix + ".mlp.experts.down_proj"]
    down_bias = tensors[layer_prefix + ".mlp.experts.down_proj_bias"]
    top_k = int(probe["experts_per_token"])

    class Router:
        def __call__(self, hidden_states):
            logits = torch.nn.functional.linear(hidden_states, router_weight.float(), None if router_bias is None else router_bias.float())
            values, indices = torch.topk(logits, top_k, dim=-1)
            weights = torch.softmax(values, dim=1, dtype=values.dtype)
            return logits, weights, indices

    router = Router()
    hidden = torch.randn(1, min(2, args.smoke_tokens), int(probe["hidden_size"]), dtype=torch.float32)
    dense_output = fake_gpt_oss_full_fused_moe_forward(
        hidden,
        router,
        gate_up.float(),
        gate_up_bias.float(),
        down.float(),
        down_bias.float(),
    )
    _, routing_weights, router_indices = router(hidden)
    selected = sorted(int(item) for item in torch.unique(router_indices).tolist())
    all_shards = build_fake_gpt_oss_expert_shards(gate_up.float(), gate_up_bias.float(), down.float(), down_bias.float())
    requested = []

    def load_selected(expert_id):
        requested.append(int(expert_id))
        return all_shards[int(expert_id)]

    adapter = GptOssSelectiveFusedMoEAdapter(
        router=None,
        expert_loader=load_selected,
        num_experts=int(probe["num_experts"]),
        top_k=top_k,
    )
    selective_output = adapter.forward(hidden, router_indices=router_indices, routing_weights=routing_weights)
    torch.testing.assert_close(selective_output, dense_output, rtol=1e-4, atol=1e-4)
    return {
        "one_layer_smoke": "selective_matches_dense",
        "downloaded_shards": shard_names,
        "layer_prefix": layer_prefix,
        "selected_experts": selected,
        "requested_experts": requested,
        "only_selected_experts_requested": sorted(set(requested)) == selected,
        "strict_full_fused_tensor_loaded": False,
        "output_shape": list(selective_output.shape),
        "output_dtype": str(selective_output.dtype),
    }


def _run_qwen35_one_layer_smoke(args, probe):
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        import torch
    except ImportError as exc:
        raise SystemExit(
            "--download-one-layer requires huggingface_hub, safetensors, and torch. "
            "Install requirements-dev.txt first."
        ) from exc

    from airllm.moe_layout_probe import QWEN35_EXPERT_SUFFIXES, QWEN35_ROUTER_SUFFIXES, load_safetensors_index
    from airllm.selective_fused_moe import Qwen35SelectiveFusedMoEAdapter

    index = load_safetensors_index(args.model, cache_dir=args.cache_dir, token=args.hf_token)
    weight_map = index.get("weight_map") or {}
    sample_layers = probe.get("sample_expected_layer_tensors") or {}
    if not sample_layers:
        raise SystemExit("No complete Qwen3.5/Qwen3.6 MoE layer was found in index metadata")
    layer_prefix = next(iter(sample_layers.keys()))
    tensor_names = [layer_prefix + suffix for suffix in (*QWEN35_ROUTER_SUFFIXES, *QWEN35_EXPERT_SUFFIXES)]
    missing = [name for name in tensor_names if name not in weight_map]
    if missing:
        raise SystemExit(f"Cannot smoke test {layer_prefix}; missing tensors: {missing}")

    local_root = Path(args.model).expanduser() if Path(args.model).expanduser().is_dir() else None
    shard_names = sorted({weight_map[name] for name in tensor_names})
    if local_root is not None:
        shard_paths = {shard: local_root / shard for shard in shard_names}
    else:
        shard_paths = {
            shard: Path(hf_hub_download(args.model, shard, cache_dir=args.cache_dir, token=args.hf_token))
            for shard in shard_names
        }

    tensor_shapes = {}
    router_name = layer_prefix + ".mlp.gate.weight"
    gate_up_name = layer_prefix + ".mlp.experts.gate_up_proj"
    down_name = layer_prefix + ".mlp.experts.down_proj"
    with safe_open(str(shard_paths[weight_map[router_name]]), framework="pt", device="cpu") as handle:
        router_weight = handle.get_tensor(router_name).float()
        tensor_shapes[router_name] = list(router_weight.shape)

    hidden = torch.randn(1, min(2, args.smoke_tokens), int(probe["hidden_size"]), dtype=torch.float32)
    flat = hidden.reshape(-1, hidden.shape[-1])
    router_logits = torch.nn.functional.linear(flat, router_weight)
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    top_k_weights, top_k_index = torch.topk(router_probs, int(probe["experts_per_token"]), dim=-1)
    top_k_weights = (top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)).to(router_logits.dtype)
    selected = sorted(int(item) for item in torch.unique(top_k_index).tolist())

    expert_shards = {}
    for expert_id in selected:
        expert_state = {}
        for name, mapped in ((gate_up_name, "gate_up_proj.weight"), (down_name, "down_proj.weight")):
            with safe_open(str(shard_paths[weight_map[name]]), framework="pt", device="cpu") as handle:
                safe_slice = handle.get_slice(name)
                tensor_shapes[name] = list(safe_slice.get_shape())
                expert_state[mapped] = safe_slice[expert_id].contiguous().float()
        expert_shards[expert_id] = expert_state

    requested = []

    def load_selected(expert_id):
        requested.append(int(expert_id))
        return expert_shards[int(expert_id)]

    adapter = Qwen35SelectiveFusedMoEAdapter(
        router=None,
        expert_loader=load_selected,
        num_experts=int(probe["num_experts"]),
        top_k=int(probe["experts_per_token"]),
    )
    selective_output = adapter.forward(flat, top_k_index, top_k_weights)

    dense_output = torch.zeros_like(flat)
    for expert_id in selected:
        token_idx, top_k_pos = torch.where(top_k_index == expert_id)
        current = flat[token_idx]
        gate, up = torch.nn.functional.linear(current, expert_shards[expert_id]["gate_up_proj.weight"]).chunk(2, dim=-1)
        out = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, expert_shards[expert_id]["down_proj.weight"])
        dense_output.index_add_(0, token_idx, out * top_k_weights[token_idx, top_k_pos, None])

    torch.testing.assert_close(selective_output, dense_output, rtol=1e-4, atol=1e-4)
    return {
        "one_layer_smoke": "qwen3_5_moe_selected_experts_match_dense_slice",
        "downloaded_shards": shard_names,
        "layer_prefix": layer_prefix,
        "tensor_shapes": tensor_shapes,
        "selected_experts": selected,
        "requested_experts": requested,
        "only_selected_experts_requested": sorted(set(requested)) == selected,
        "strict_full_fused_tensor_loaded": False,
        "shared_expert_remains_dense": bool(probe.get("has_shared_expert")),
        "output_shape": list(selective_output.shape),
        "output_dtype": str(selective_output.dtype),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face repo id or local checkpoint directory")
    parser.add_argument("--no-full-download", action="store_true", help="Only download config/index metadata. This is the default.")
    parser.add_argument("--cache-dir")
    parser.add_argument("--hf-token")
    parser.add_argument("--verify-manifest", action="store_true", help="Fail if strict GPT-OSS manifest checks do not pass")
    parser.add_argument("--download-one-layer", action="store_true", help="Optional GPT-OSS 20B one-layer smoke test; downloads needed shard files")
    parser.add_argument("--download-one-expert", action="store_true", help="Optional GPT-OSS 20B one-expert MXFP4 smoke; downloads needed shard files")
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--confirm-download", action="store_true", help="Required before downloading real checkpoint shard files")
    parser.add_argument("--smoke-tokens", type=int, default=2)
    args = parser.parse_args()

    probe = probe_moe_layout(args.model, cache_dir=args.cache_dir, token=args.hf_token)
    if args.download_one_expert:
        if not args.confirm_download:
            raise SystemExit("--download-one-expert requires --confirm-download because it downloads checkpoint shard files")
        probe["one_expert_mxfp4_smoke_test"] = download_gpt_oss_one_expert_mxfp4_smoke(
            args.model,
            layer_id=args.layer_id,
            expert_id=args.expert_id,
            cache_dir=args.cache_dir,
            token=args.hf_token,
        )
    if args.download_one_layer:
        if not args.confirm_download:
            raise SystemExit("--download-one-layer requires --confirm-download because it downloads checkpoint shard files")
        probe["one_layer_smoke_test"] = _run_one_layer_smoke(args, probe)

    print(json.dumps(probe, indent=2, sort_keys=True))

    if args.verify_manifest and not probe["manifest_verification"]["manifest_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
