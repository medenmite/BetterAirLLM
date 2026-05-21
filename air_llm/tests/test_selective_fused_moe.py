import unittest

import torch

from ..betterairllm.selective_fused_moe import (
    FakeFusedMoEAdapter,
    build_fake_fused_expert_shards,
    fake_full_expert_bytes,
    fake_full_fused_moe_forward,
)


class TinyFakeFusedMoE(torch.nn.Module):
    def __init__(self, hidden_dim=8, intermediate_dim=12, num_experts=5, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        self.gate_up_proj = torch.nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))
        self.down_proj = torch.nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(123)
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.15)
        torch.nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.08)
        torch.nn.init.normal_(self.down_proj, mean=0.0, std=0.08)

    def forward(self, hidden_states):
        return fake_full_fused_moe_forward(
            hidden_states,
            self.router,
            self.gate_up_proj,
            self.down_proj,
            self.top_k,
        )

    def selective_forward(self, hidden_states, cache_size=0):
        shards = build_fake_fused_expert_shards(self.gate_up_proj, self.down_proj)
        adapter = FakeFusedMoEAdapter(
            router=self.router,
            expert_loader=lambda expert_id: shards[int(expert_id)],
            num_experts=self.num_experts,
            top_k=self.top_k,
            cache_size=cache_size,
        )
        return adapter.forward(hidden_states), adapter


class TestSelectiveFusedMoE(unittest.TestCase):
    def test_fake_selective_fused_output_matches_full_fused_output(self):
        torch.manual_seed(321)
        block = TinyFakeFusedMoE()
        hidden_states = torch.randn(2, 7, block.hidden_dim)

        full_output = block(hidden_states)
        selective_output, _ = block.selective_forward(hidden_states)

        torch.testing.assert_close(selective_output, full_output, rtol=1e-5, atol=1e-6)

    def test_fake_selective_fused_loads_less_than_full_fused(self):
        torch.manual_seed(321)
        block = TinyFakeFusedMoE(num_experts=8, top_k=2)
        hidden_states = torch.randn(1, 4, block.hidden_dim)
        _, adapter = block.selective_forward(hidden_states)

        full_bytes = fake_full_expert_bytes(block.gate_up_proj, block.down_proj)
        selective_bytes = adapter.stats["expert_load_bytes"]

        self.assertLess(selective_bytes, full_bytes)


if __name__ == "__main__":
    unittest.main()
