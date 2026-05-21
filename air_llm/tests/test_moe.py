import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ..betterairllm.auto_model import AutoModel
from ..betterairllm.utils import split_moe_layer_state_dict


class TestMoEUtilities(unittest.TestCase):
    def test_split_moe_layer_state_dict_keeps_dense_and_groups_experts(self):
        tensor = object()
        state_dict = {
            "model.layers.0.input_layernorm.weight": tensor,
            "model.layers.0.block_sparse_moe.gate.weight": tensor,
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": tensor,
            "model.layers.0.block_sparse_moe.experts.0.w2.weight": tensor,
            "model.layers.0.block_sparse_moe.experts.1.w1.weight": tensor,
            "model.layers.0.block_sparse_moe.experts.shared_router.w1.weight": tensor,
            "model.layers.0.block_sparse_moe.shared_experts.w1.weight": tensor,
        }

        dense, experts = split_moe_layer_state_dict(state_dict)

        self.assertIn("model.layers.0.input_layernorm.weight", dense)
        self.assertIn("model.layers.0.block_sparse_moe.gate.weight", dense)
        self.assertIn("model.layers.0.block_sparse_moe.shared_experts.w1.weight", dense)
        self.assertNotIn("model.layers.0.block_sparse_moe.experts.0.w1.weight", dense)
        self.assertEqual(
            sorted(experts.keys()),
            [
                "model.layers.0.block_sparse_moe.experts.0",
                "model.layers.0.block_sparse_moe.experts.1",
                "model.layers.0.block_sparse_moe.experts.shared_router",
            ],
        )

    def test_auto_model_routes_generic_moe_configs_to_moe_runtime(self):
        config = SimpleNamespace(
            architectures=["Qwen2MoeForCausalLM"],
            model_type="qwen2_moe",
            num_experts=64,
        )

        with patch("air_llm.betterairllm.auto_model.AutoConfig.from_pretrained", return_value=config):
            module, cls = AutoModel.get_module_class("local-moe-model")

        self.assertEqual(module, "betterairllm")
        self.assertEqual(cls, "BetterAirLLMMoE")

    def test_split_moe_layer_state_dict_splits_fused_expert_tensors(self):
        state_dict = {
            "model.layers.0.mlp.experts.gate_up_proj.weight": torch.arange(24).reshape(3, 2, 4),
            "model.layers.0.mlp.router.weight": torch.ones(3, 4),
        }

        dense, experts = split_moe_layer_state_dict(state_dict, num_experts=3)

        self.assertIn("model.layers.0.mlp.router.weight", dense)
        self.assertIn("model.layers.0.mlp.experts.gate_up_proj.weight", dense)
        self.assertEqual(
            sorted(experts.keys()),
            [
                "model.layers.0.mlp.experts.0",
                "model.layers.0.mlp.experts.1",
                "model.layers.0.mlp.experts.2",
            ],
        )
        self.assertTrue(
            torch.equal(
                experts["model.layers.0.mlp.experts.1"]["model.layers.0.mlp.experts.1.gate_up_proj.weight"],
                state_dict["model.layers.0.mlp.experts.gate_up_proj.weight"][1],
            )
        )

    def test_split_moe_layer_state_dict_can_drop_fused_tensor_for_strict_adapter(self):
        state_dict = {
            "model.layers.0.mlp.experts.gate_up_proj.weight": torch.arange(24).reshape(3, 2, 4),
            "model.layers.0.mlp.router.weight": torch.ones(3, 4),
        }

        dense, experts, metadata = split_moe_layer_state_dict(
            state_dict,
            num_experts=3,
            return_metadata=True,
            drop_fused_from_dense=True,
        )

        self.assertIn("model.layers.0.mlp.router.weight", dense)
        self.assertNotIn("model.layers.0.mlp.experts.gate_up_proj.weight", dense)
        self.assertEqual(len(experts), 3)
        self.assertEqual(
            metadata["fused_tensor_keys"],
            ["model.layers.0.mlp.experts.gate_up_proj.weight"],
        )


if __name__ == "__main__":
    unittest.main()
