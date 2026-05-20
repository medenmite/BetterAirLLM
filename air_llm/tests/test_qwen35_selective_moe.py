import json

import torch
from safetensors.torch import save_file

from airllm.moe_layout_probe import build_manifest_verification, probe_qwen35_layout
from airllm.moe_split_builder import IncrementalMoESplitBuilder
from airllm.selective_fused_moe import Qwen35SelectiveFusedMoEAdapter
from airllm.utils import split_moe_layer_state_dict


def _dense_qwen_routed(hidden, gate_up, down, top_k_index, top_k_weights):
    output = torch.zeros_like(hidden)
    for expert_id in torch.unique(top_k_index).tolist():
        expert_id = int(expert_id)
        token_idx, top_k_pos = torch.where(top_k_index == expert_id)
        current = hidden[token_idx]
        gate, up = torch.nn.functional.linear(current, gate_up[expert_id]).chunk(2, dim=-1)
        expert_output = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down[expert_id])
        output.index_add_(0, token_idx, expert_output * top_k_weights[token_idx, top_k_pos, None])
    return output


def test_qwen35_selective_adapter_matches_dense_and_requests_only_selected():
    torch.manual_seed(0)
    hidden = torch.randn(5, 4)
    gate_up = torch.randn(4, 6, 4)
    down = torch.randn(4, 4, 3)
    top_k_index = torch.tensor([[0, 2], [2, 3], [0, 3], [2, 0], [3, 2]])
    top_k_weights = torch.rand(5, 2)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    selected = sorted(torch.unique(top_k_index).tolist())
    requested = []

    def load_expert(expert_id):
        requested.append(int(expert_id))
        return {
            "gate_up_proj.weight": gate_up[int(expert_id)].clone(),
            "down_proj.weight": down[int(expert_id)].clone(),
        }

    adapter = Qwen35SelectiveFusedMoEAdapter(
        router=None,
        expert_loader=load_expert,
        num_experts=4,
        top_k=2,
    )

    expected = _dense_qwen_routed(hidden, gate_up, down, top_k_index, top_k_weights)
    actual = adapter.forward(hidden, top_k_index, top_k_weights)

    torch.testing.assert_close(actual, expected)
    assert sorted(set(requested)) == selected
    assert 1 not in requested


def test_qwen35_shared_expert_remains_outside_selective_routed_path():
    torch.manual_seed(1)
    hidden = torch.randn(3, 4)
    gate_up = torch.randn(2, 6, 4)
    down = torch.randn(2, 4, 3)
    top_k_index = torch.tensor([[0], [1], [0]])
    top_k_weights = torch.ones(3, 1)
    shared_gate = torch.randn(1, 4)
    shared_up = torch.randn(3, 4)
    shared_down = torch.randn(4, 3)

    def load_expert(expert_id):
        return {
            "gate_up_proj.weight": gate_up[int(expert_id)],
            "down_proj.weight": down[int(expert_id)],
        }

    adapter = Qwen35SelectiveFusedMoEAdapter(
        router=None,
        expert_loader=load_expert,
        num_experts=2,
        top_k=1,
    )
    routed = adapter.forward(hidden, top_k_index, top_k_weights)
    shared = (
        torch.sigmoid(torch.nn.functional.linear(hidden, shared_gate))
        * torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(hidden, shared_up)), shared_down)
    )

    expected_final = _dense_qwen_routed(hidden, gate_up, down, top_k_index, top_k_weights) + shared
    torch.testing.assert_close(routed + shared, expected_final)


def test_qwen35_split_drops_routed_fused_experts_but_keeps_shared_expert():
    layer_state = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 6, 4),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(2, 4, 3),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(3, 4),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(3, 4),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(4, 3),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": torch.randn(1, 4),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(2, 4),
    }

    dense, experts, metadata = split_moe_layer_state_dict(
        layer_state,
        num_experts=2,
        return_metadata=True,
        drop_fused_from_dense=True,
    )

    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in dense
    assert "model.language_model.layers.0.mlp.experts.down_proj" not in dense
    assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" in dense
    assert sorted(experts) == [
        "model.language_model.layers.0.mlp.experts.0",
        "model.language_model.layers.0.mlp.experts.1",
    ]
    assert len(metadata["fused_tensor_keys"]) == 2


def test_qwen35_probe_manifest_reports_strict_selective_runtime(tmp_path):
    config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 1,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "hidden_size": 4,
            "moe_intermediate_size": 3,
        },
    }
    weight_map = {
        "model.language_model.layers.0.mlp.gate.weight": "model-00001-of-00001.safetensors",
        "model.language_model.layers.0.mlp.experts.gate_up_proj": "model-00001-of-00001.safetensors",
        "model.language_model.layers.0.mlp.experts.down_proj": "model-00001-of-00001.safetensors",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "model-00001-of-00001.safetensors",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": "model-00001-of-00001.safetensors",
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "model-00001-of-00001.safetensors",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": "model-00001-of-00001.safetensors",
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    probe = probe_qwen35_layout(str(tmp_path))
    manifest = build_manifest_verification(config, weight_map, adapter_name="qwen3_5_moe")

    assert probe["num_experts"] == 256
    assert probe["experts_per_token"] == 8
    assert probe["moe_intermediate_size"] == 3
    assert probe["has_shared_expert"] is True
    assert probe["strict_selective_runtime_can_enable"] is True
    assert manifest["selective_fused_runtime"] is True
    assert manifest["dense_contains_full_fused_experts"] is False
    assert manifest["adapter_name"] == "qwen3_5_moe"


def test_qwen35_builder_direct_slice_loads_single_selected_expert(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    split = tmp_path / "split"
    checkpoint.mkdir()
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(2, 4),
    }
    save_file(tensors, checkpoint / "model-00001-of-00001.safetensors")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 1},
            "weight_map": {key: "model-00001-of-00001.safetensors" for key in tensors},
        }),
        encoding="utf-8",
    )
    builder = IncrementalMoESplitBuilder(
        checkpoint,
        split,
        layer_names={
            "embed": "model.language_model.embed_tokens",
            "layer_prefix": "model.language_model.layers",
            "norm": "model.language_model.norm",
            "lm_head": "lm_head",
        },
        strict_mode=True,
        adapter_name="qwen3_5_moe",
    )

    state = builder.load_qwen35_expert_direct("model.language_model.layers.0.mlp.experts.1")

    assert sorted(state) == ["down_proj.weight", "gate_up_proj.weight"]
    torch.testing.assert_close(state["gate_up_proj.weight"], tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"][1])
    assert builder.status()["direct_slice_expert_loads_this_run"] == 1


def test_qwen35_builder_batch_direct_slice_loads_selected_experts_once(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    split = tmp_path / "split"
    checkpoint.mkdir()
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 1,
            "num_experts": 3,
            "num_experts_per_tok": 2,
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.arange(3 * 6 * 4, dtype=torch.float32).reshape(3, 6, 4),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.arange(3 * 4 * 3, dtype=torch.float32).reshape(3, 4, 3),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(3, 4),
    }
    save_file(tensors, checkpoint / "model-00001-of-00001.safetensors")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 1},
            "weight_map": {key: "model-00001-of-00001.safetensors" for key in tensors},
        }),
        encoding="utf-8",
    )
    builder = IncrementalMoESplitBuilder(
        checkpoint,
        split,
        layer_names={
            "embed": "model.language_model.embed_tokens",
            "layer_prefix": "model.language_model.layers",
            "norm": "model.language_model.norm",
            "lm_head": "lm_head",
        },
        strict_mode=True,
        adapter_name="qwen3_5_moe",
    )

    states = builder.load_qwen35_experts_direct([
        "model.language_model.layers.0.mlp.experts.0",
        "model.language_model.layers.0.mlp.experts.2",
    ])

    assert sorted(states) == [
        "model.language_model.layers.0.mlp.experts.0",
        "model.language_model.layers.0.mlp.experts.2",
    ]
    torch.testing.assert_close(
        states["model.language_model.layers.0.mlp.experts.2"]["down_proj.weight"],
        tensors["model.language_model.layers.0.mlp.experts.down_proj"][2],
    )
    assert builder.status()["direct_slice_expert_loads_this_run"] == 2
    assert builder.progress["layer_timings"]["model.language_model.layers.0."]["selected_experts"]["2"]["mode"] == "direct_slice_batch"
