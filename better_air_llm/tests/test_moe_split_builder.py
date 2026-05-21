import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from betterairllm.moe_split_builder import IncrementalMoESplitBuilder


class IncrementalMoESplitBuilderTest(unittest.TestCase):
    def _make_checkpoint(self, root):
        checkpoint = Path(root)
        config = {
            "model_type": "gpt_oss",
            "num_hidden_layers": 2,
            "num_local_experts": 3,
            "num_experts_per_tok": 1,
            "hidden_size": 4,
            "intermediate_size": 4,
        }
        (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
        tensors = {
            "model.embed_tokens.weight": torch.randn(8, 4),
            "model.layers.0.input_layernorm.weight": torch.randn(4),
            "model.layers.0.mlp.router.weight": torch.randn(3, 4),
            "model.layers.0.mlp.experts.gate_up_proj_blocks": torch.zeros(3, 8, 1, 16, dtype=torch.uint8),
            "model.layers.0.mlp.experts.gate_up_proj_scales": torch.full((3, 8, 1), 127, dtype=torch.uint8),
            "model.layers.0.mlp.experts.gate_up_proj_bias": torch.randn(3, 8),
            "model.layers.0.mlp.experts.down_proj_blocks": torch.zeros(3, 4, 1, 16, dtype=torch.uint8),
            "model.layers.0.mlp.experts.down_proj_scales": torch.full((3, 4, 1), 127, dtype=torch.uint8),
            "model.layers.0.mlp.experts.down_proj_bias": torch.randn(3, 4),
            "model.layers.1.input_layernorm.weight": torch.randn(4),
            "model.layers.1.mlp.router.weight": torch.randn(3, 4),
            "model.layers.1.mlp.experts.gate_up_proj_blocks": torch.zeros(3, 8, 1, 16, dtype=torch.uint8),
            "model.layers.1.mlp.experts.gate_up_proj_scales": torch.full((3, 8, 1), 127, dtype=torch.uint8),
            "model.layers.1.mlp.experts.gate_up_proj_bias": torch.randn(3, 8),
            "model.layers.1.mlp.experts.down_proj_blocks": torch.zeros(3, 4, 1, 16, dtype=torch.uint8),
            "model.layers.1.mlp.experts.down_proj_scales": torch.full((3, 4, 1), 127, dtype=torch.uint8),
            "model.layers.1.mlp.experts.down_proj_bias": torch.randn(3, 4),
            "model.norm.weight": torch.randn(4),
            "lm_head.weight": torch.randn(8, 4),
        }
        save_file(tensors, checkpoint / "model-00000-of-00001.safetensors")
        index = {"metadata": {"total_size": (checkpoint / "model-00000-of-00001.safetensors").stat().st_size},
                 "weight_map": {key: "model-00000-of-00001.safetensors" for key in tensors}}
        (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        return checkpoint

    def test_incremental_resume_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._make_checkpoint(tmp)
            split_dir = checkpoint / "splitted_model.moe"
            builder = IncrementalMoESplitBuilder(
                checkpoint,
                split_dir,
                strict_mode=True,
                adapter_name="gpt_oss_mxfp4_reference",
            )
            builder.ensure_layer("model.layers.0")
            first_status = builder.status()
            self.assertGreaterEqual(first_status["completed_layers"], 1)
            self.assertTrue((split_dir / "progress.json").exists())
            self.assertTrue((split_dir / "model.layers.0.safetensors").exists())
            self.assertTrue((split_dir / "model.layers.0.mlp.experts.0.safetensors").exists())

            resumed = IncrementalMoESplitBuilder(
                checkpoint,
                split_dir,
                strict_mode=True,
                adapter_name="gpt_oss_mxfp4_reference",
            )
            before = resumed.status()["completed_layers"]
            resumed.ensure_layer("model.layers.0")
            self.assertEqual(resumed.status()["completed_layers"], before)
            resumed.ensure_layer("model.layers.1")
            self.assertTrue((split_dir / "moe_expert_index.json").exists())
            manifest = json.loads((split_dir / "moe_expert_index.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["selective_fused_runtime"])
            self.assertFalse(manifest["dense_contains_full_fused_experts"])
            resumed.verify_layer("model.layers.0")

    def test_dense_ready_and_selected_expert_only_materialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._make_checkpoint(tmp)
            split_dir = checkpoint / "splitted_model.moe"
            builder = IncrementalMoESplitBuilder(
                checkpoint,
                split_dir,
                strict_mode=True,
                adapter_name="gpt_oss_mxfp4_reference",
            )

            builder.ensure_layer_dense_ready("model.layers.0")
            self.assertTrue((split_dir / "model.layers.0.safetensors").exists())
            self.assertFalse((split_dir / "model.layers.0.mlp.experts.0.safetensors").exists())
            verified = builder.verify_layer("model.layers.0")
            self.assertTrue(verified["valid"])
            self.assertTrue(all(".experts." not in key for key in verified["tensors"]))

            builder.ensure_selected_experts_ready("model.layers.0", [1])
            self.assertFalse((split_dir / "model.layers.0.mlp.experts.0.safetensors").exists())
            self.assertTrue((split_dir / "model.layers.0.mlp.experts.1.safetensors").exists())
            self.assertFalse((split_dir / "model.layers.0.mlp.experts.2.safetensors").exists())
            status = builder.status()
            self.assertEqual(status["selected_expert_builds_this_run"], 1)
            self.assertEqual(status["unused_expert_builds_this_run"], 0)

            direct = builder.load_gpt_oss_expert_direct("model.layers.0.mlp.experts.2")
            self.assertIn("gate_up_proj_blocks", direct)
            self.assertFalse((split_dir / "model.layers.0.mlp.experts.2.safetensors").exists())
            status = builder.status()
            self.assertEqual(status["direct_slice_expert_loads_this_run"], 1)
            self.assertEqual(status["unused_expert_builds_this_run"], 0)


if __name__ == "__main__":
    unittest.main()
