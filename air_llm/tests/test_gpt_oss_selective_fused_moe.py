import unittest

import torch
import torch.nn.functional as F

from ..airllm.selective_fused_moe import (
    GptOssSelectiveFusedMoEAdapter,
    build_fake_gpt_oss_expert_shards,
    fake_gpt_oss_full_expert_bytes,
    fake_gpt_oss_full_fused_moe_forward,
)
from ..airllm.airllm_moe import BetterAirLLMMoE
from ..airllm.utils import split_moe_layer_state_dict


class TinyGptOssRouter(torch.nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.weight = torch.nn.Parameter(torch.zeros(num_experts, hidden_dim))
        self.bias = torch.nn.Parameter(torch.zeros(num_experts))

    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_scores = torch.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        return router_logits, router_scores, router_indices


class TinyGptOssFusedMoE(torch.nn.Module):
    def __init__(self, hidden_dim=8, intermediate_dim=12, num_experts=8, top_k=4, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TinyGptOssRouter(hidden_dim, num_experts, top_k)
        self.gate_up_proj = torch.nn.Parameter(torch.empty(num_experts, hidden_dim, 2 * intermediate_dim))
        self.gate_up_proj_bias = torch.nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim))
        self.down_proj = torch.nn.Parameter(torch.empty(num_experts, intermediate_dim, hidden_dim))
        self.down_proj_bias = torch.nn.Parameter(torch.empty(num_experts, hidden_dim))
        self.reset_parameters()
        self.to(dtype=dtype)

    def reset_parameters(self):
        torch.manual_seed(911)
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.2)
        torch.nn.init.normal_(self.router.bias, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.05)
        torch.nn.init.normal_(self.gate_up_proj_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.down_proj, mean=0.0, std=0.05)
        torch.nn.init.normal_(self.down_proj_bias, mean=0.0, std=0.01)

    def forward(self, hidden_states):
        return fake_gpt_oss_full_fused_moe_forward(
            hidden_states,
            self.router,
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
        )

    def adapter(self, cache_size=0, loader_hook=None, expert_execution_mode="grouped_by_expert"):
        shards = build_fake_gpt_oss_expert_shards(
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
        )

        def load(expert_id):
            if loader_hook is not None:
                loader_hook(int(expert_id))
            return shards[int(expert_id)]

        return GptOssSelectiveFusedMoEAdapter(
            router=self.router,
            expert_loader=load,
            num_experts=self.num_experts,
            top_k=self.top_k,
            cache_size=cache_size,
            expert_execution_mode=expert_execution_mode,
        )


class TestGptOssSelectiveFusedMoE(unittest.TestCase):
    def assert_selective_matches_dense(self, batch, seq, dtype=torch.float32):
        block = TinyGptOssFusedMoE(dtype=dtype)
        hidden_states = torch.randn(batch, seq, block.hidden_dim, dtype=dtype)
        dense = block(hidden_states)
        adapter = block.adapter()
        selective = adapter.forward(hidden_states)
        tolerance = 2e-2 if dtype == torch.float16 else 1e-5
        torch.testing.assert_close(selective, dense, rtol=tolerance, atol=tolerance)
        return block, adapter, hidden_states

    def test_float32_batch_one_matches_dense_top4(self):
        self.assert_selective_matches_dense(batch=1, seq=5, dtype=torch.float32)

    def test_float32_batch_gt_one_matches_dense_top4(self):
        self.assert_selective_matches_dense(batch=3, seq=4, dtype=torch.float32)

    def test_float16_matches_dense_when_supported(self):
        self.assert_selective_matches_dense(batch=1, seq=3, dtype=torch.float16)

    def test_multiple_tokens_and_repeated_experts(self):
        block = TinyGptOssFusedMoE(num_experts=6, top_k=4)
        with torch.no_grad():
            block.router.weight.zero_()
            block.router.bias[:] = torch.tensor([5.0, 4.0, 3.0, 2.0, -1.0, -2.0])
        hidden_states = torch.randn(2, 3, block.hidden_dim)
        adapter = block.adapter(cache_size=8)
        selective = adapter.forward(hidden_states)
        dense = block(hidden_states)

        torch.testing.assert_close(selective, dense, rtol=1e-5, atol=1e-6)
        self.assertEqual(adapter.stats["expert_load_calls"], 4)
        self.assertEqual(adapter.stats["selected_expert_slots"], 2 * 3 * 4)
        self.assertEqual(adapter.stats["run_expert_calls"], 4)
        self.assertEqual(adapter.stats["grouped_expert_calls"], 4)

    def test_per_token_and_grouped_by_expert_outputs_match(self):
        block = TinyGptOssFusedMoE(num_experts=6, top_k=4)
        with torch.no_grad():
            block.router.weight.zero_()
            block.router.bias[:] = torch.tensor([5.0, 4.0, 3.0, 2.0, -1.0, -2.0])
        hidden_states = torch.randn(2, 3, block.hidden_dim)

        grouped = block.adapter(cache_size=8, expert_execution_mode="grouped_by_expert")
        per_token = block.adapter(cache_size=8, expert_execution_mode="per_token")
        grouped_output = grouped.forward(hidden_states)
        per_token_output = per_token.forward(hidden_states)

        torch.testing.assert_close(grouped_output, per_token_output, rtol=1e-5, atol=1e-6)
        self.assertLess(grouped.stats["run_expert_calls"], per_token.stats["run_expert_calls"])
        self.assertEqual(per_token.stats["run_expert_calls"], per_token.stats["selected_expert_slots"])

    def test_only_selected_experts_are_loaded(self):
        block = TinyGptOssFusedMoE(num_experts=12, top_k=4)
        loaded = []
        hidden_states = torch.randn(1, 2, block.hidden_dim)
        adapter = block.adapter(loader_hook=loaded.append)
        _, _, selected = block.router(hidden_states.reshape(-1, block.hidden_dim))
        adapter.forward(hidden_states)

        self.assertEqual(set(loaded), set(int(x) for x in torch.unique(selected).tolist()))
        self.assertLess(len(loaded), block.num_experts)

    def test_resident_cache_hit_path_and_lru_eviction(self):
        block = TinyGptOssFusedMoE(num_experts=8, top_k=4)
        with torch.no_grad():
            block.router.weight.zero_()
            block.router.bias[:] = torch.tensor([8.0, 7.0, 6.0, 5.0, -1.0, -2.0, -3.0, -4.0])
        hidden_states = torch.randn(1, 2, block.hidden_dim)
        adapter = block.adapter(cache_size=4)
        adapter.forward(hidden_states)
        adapter.forward(hidden_states)
        self.assertGreater(adapter.stats["expert_cache_hits"], 0)
        self.assertEqual(len(adapter._expert_cache), 4)

        with torch.no_grad():
            block.router.bias[:] = torch.tensor([-1.0, -2.0, -3.0, -4.0, 8.0, 7.0, 6.0, 5.0])
        adapter.forward(hidden_states)
        self.assertEqual(len(adapter._expert_cache), 4)
        self.assertGreater(adapter.stats["expert_cache_misses"], 4)

    def test_disk_path_reloads_without_resident_cache(self):
        block = TinyGptOssFusedMoE(num_experts=8, top_k=4)
        with torch.no_grad():
            block.router.weight.zero_()
            block.router.bias[:] = torch.tensor([8.0, 7.0, 6.0, 5.0, -1.0, -2.0, -3.0, -4.0])
        hidden_states = torch.randn(1, 1, block.hidden_dim)
        adapter = block.adapter(cache_size=0)
        adapter.forward(hidden_states)
        first_loads = adapter.stats["expert_load_calls"]
        adapter.forward(hidden_states)
        self.assertEqual(adapter.stats["expert_load_calls"], first_loads * 2)
        self.assertEqual(adapter.stats["expert_cache_hits"], 0)

    def test_airllm_cpu_expert_cache_hit_path(self):
        model = BetterAirLLMMoE.__new__(BetterAirLLMMoE)
        model._cpu_expert_cache = {}
        model._cpu_expert_bytes = 0
        model._cpu_expert_budget_bytes = 1024 * 1024
        model._cpu_expert_cache_hits = 0
        model._cpu_expert_cache_misses = 0
        calls = []
        state_dict = {"x": torch.ones(4)}

        def load_layer_to_cpu(shard_name):
            calls.append(shard_name)
            return state_dict

        model.load_layer_to_cpu = load_layer_to_cpu
        model._load_expert_state_dict("layer.experts.0")
        model._load_expert_state_dict("layer.experts.0")

        self.assertEqual(calls, ["layer.experts.0"])
        self.assertEqual(model._cpu_expert_cache_hits, 1)
        self.assertEqual(model._cpu_expert_cache_misses, 1)

    def test_strict_split_removes_full_fused_tensors(self):
        block = TinyGptOssFusedMoE()
        state_dict = {
            "model.layers.0.mlp.experts.gate_up_proj": block.gate_up_proj.detach(),
            "model.layers.0.mlp.experts.gate_up_proj_bias": block.gate_up_proj_bias.detach(),
            "model.layers.0.mlp.experts.down_proj": block.down_proj.detach(),
            "model.layers.0.mlp.experts.down_proj_bias": block.down_proj_bias.detach(),
            "model.layers.0.mlp.router.weight": block.router.weight.detach(),
        }
        dense, experts, metadata = split_moe_layer_state_dict(
            state_dict,
            num_experts=block.num_experts,
            return_metadata=True,
            drop_fused_from_dense=True,
        )

        self.assertNotIn("model.layers.0.mlp.experts.gate_up_proj", dense)
        self.assertIn("model.layers.0.mlp.router.weight", dense)
        self.assertEqual(len(experts), block.num_experts)
        self.assertEqual(len(metadata["fused_tensor_keys"]), 4)

    def test_selective_bytes_lower_than_dense(self):
        block, adapter, hidden_states = self.assert_selective_matches_dense(batch=1, seq=2)
        dense_bytes = fake_gpt_oss_full_expert_bytes(
            block.gate_up_proj,
            block.gate_up_proj_bias,
            block.down_proj,
            block.down_proj_bias,
        )
        self.assertLess(adapter.stats["expert_load_bytes"], dense_bytes)


if __name__ == "__main__":
    unittest.main()
