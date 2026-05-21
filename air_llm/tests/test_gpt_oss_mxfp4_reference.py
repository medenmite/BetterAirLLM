import unittest
import sys
from pathlib import Path

import torch

AIR_LLM_ROOT = Path(__file__).resolve().parents[1]
if str(AIR_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(AIR_LLM_ROOT))

from betterairllm.gpt_oss_mxfp4 import (
    FP4_VALUES,
    dequantize_mxfp4_projection,
    mxfp4_state_nbytes,
    pack_dense_to_mxfp4_exact,
    run_gpt_oss_selected_expert_reference,
)
from betterairllm.selective_fused_moe import GptOssSelectiveFusedMoEAdapter


class FixedTop4Router:
    def __init__(self, indices):
        self.indices = torch.tensor(indices, dtype=torch.long)

    def __call__(self, hidden_states):
        token_count = hidden_states.shape[0]
        indices = self.indices[:token_count].to(hidden_states.device)
        weights = torch.full(indices.shape, 0.25, dtype=hidden_states.dtype, device=hidden_states.device)
        logits = torch.zeros(token_count, int(indices.max().item()) + 1, dtype=hidden_states.dtype, device=hidden_states.device)
        return logits, weights, indices


def _make_representable_dense(shape, offset=0):
    values = torch.tensor(FP4_VALUES, dtype=torch.float32)
    total = 1
    for dim in shape:
        total *= dim
    data = values[(torch.arange(total) + offset) % len(values)]
    return (data.reshape(shape) * 0.125).contiguous()


def _make_packed_case(batch_size=1):
    torch.manual_seed(44)
    num_experts = 6
    hidden_dim = 32
    intermediate_dim = 32
    gate_up_dense = _make_representable_dense((num_experts, hidden_dim, 2 * intermediate_dim), offset=1)
    down_dense = _make_representable_dense((num_experts, intermediate_dim, hidden_dim), offset=7)
    gate_up_blocks, gate_up_scales = pack_dense_to_mxfp4_exact(gate_up_dense, scale_exponent=-3)
    down_blocks, down_scales = pack_dense_to_mxfp4_exact(down_dense, scale_exponent=-3)
    gate_up_bias = torch.randn(num_experts, 2 * intermediate_dim) * 0.01
    down_bias = torch.randn(num_experts, hidden_dim) * 0.01
    hidden_states = torch.randn(batch_size, 3, hidden_dim) * 0.1
    router = FixedTop4Router(
        [
            [0, 1, 2, 3],
            [1, 1, 4, 5],
            [2, 3, 3, 0],
            [5, 4, 1, 0],
            [0, 0, 2, 2],
            [3, 4, 5, 1],
        ]
    )
    dense_shards = {}
    packed_shards = {}
    for expert_id in range(num_experts):
        dense_shards[expert_id] = {
            "gate_up_proj": gate_up_dense[expert_id].contiguous(),
            "gate_up_proj_bias": gate_up_bias[expert_id].contiguous(),
            "down_proj": down_dense[expert_id].contiguous(),
            "down_proj_bias": down_bias[expert_id].contiguous(),
        }
        packed_shards[expert_id] = {
            "gate_up_proj_blocks": gate_up_blocks[expert_id].contiguous(),
            "gate_up_proj_scales": gate_up_scales[expert_id].contiguous(),
            "gate_up_proj_bias": gate_up_bias[expert_id].contiguous(),
            "down_proj_blocks": down_blocks[expert_id].contiguous(),
            "down_proj_scales": down_scales[expert_id].contiguous(),
            "down_proj_bias": down_bias[expert_id].contiguous(),
        }
    return router, hidden_states, dense_shards, packed_shards


def _dense_forward(router, hidden_states, shards):
    original_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, original_shape[-1])
    _, routing_weights, router_indices = router(flat)
    output = torch.zeros_like(flat)
    for expert_id in torch.unique(router_indices).tolist():
        expert_id = int(expert_id)
        token_idx, top_k_pos = torch.where(router_indices == expert_id)
        state = shards[expert_id]
        expert_input = flat[token_idx]
        gate_up = expert_input @ state["gate_up_proj"].to(expert_input.dtype) + state["gate_up_proj_bias"].to(expert_input.dtype)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=7.0)
        up = up.clamp(min=-7.0, max=7.0)
        glu = gate * torch.sigmoid(gate * 1.702)
        out = ((up + 1) * glu) @ state["down_proj"].to(expert_input.dtype) + state["down_proj_bias"].to(expert_input.dtype)
        output.index_add_(0, token_idx, out * routing_weights[token_idx, top_k_pos, None])
    return output.reshape(original_shape)


class GptOssMxfp4ReferenceTest(unittest.TestCase):
    def test_dequant_projection_round_trip(self):
        dense = _make_representable_dense((2, 32, 64), offset=3)
        blocks, scales = pack_dense_to_mxfp4_exact(dense, scale_exponent=-3)
        restored = dequantize_mxfp4_projection(blocks, scales, dtype=torch.float32)
        torch.testing.assert_close(restored, dense, rtol=0, atol=0)

    def test_selected_expert_reference_matches_dense(self):
        _, hidden_states, dense_shards, packed_shards = _make_packed_case()
        expert_input = hidden_states.reshape(-1, hidden_states.shape[-1])[:2]
        packed_output = run_gpt_oss_selected_expert_reference(
            packed_shards[0],
            expert_input,
            compute_dtype=torch.float32,
        )
        dense_output = _dense_forward(
            FixedTop4Router([[0, 0, 0, 0], [0, 0, 0, 0]]),
            expert_input.reshape(1, 2, -1),
            dense_shards,
        ).reshape(2, -1)
        torch.testing.assert_close(packed_output, dense_output, rtol=1e-5, atol=1e-6)

    def test_fake_packed_selective_batch_one_matches_dense(self):
        router, hidden_states, dense_shards, packed_shards = _make_packed_case(batch_size=1)
        requested = []

        def load_expert(expert_id):
            requested.append(int(expert_id))
            return packed_shards[int(expert_id)]

        adapter = GptOssSelectiveFusedMoEAdapter(router, load_expert, num_experts=6, top_k=4)
        actual = adapter.forward(hidden_states)
        expected = _dense_forward(router, hidden_states, dense_shards)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(actual.shape, hidden_states.shape)
        self.assertEqual(actual.dtype, hidden_states.dtype)
        self.assertEqual(sorted(set(requested)), [0, 1, 2, 3, 4, 5])

    def test_fake_packed_selective_batch_gt_one_repeated_experts(self):
        router, hidden_states, dense_shards, packed_shards = _make_packed_case(batch_size=2)
        requested = []
        adapter = GptOssSelectiveFusedMoEAdapter(
            router,
            lambda expert_id: requested.append(int(expert_id)) or packed_shards[int(expert_id)],
            num_experts=6,
            top_k=4,
        )
        actual = adapter.forward(hidden_states)
        expected = _dense_forward(router, hidden_states, dense_shards)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertLess(sum(mxfp4_state_nbytes(packed_shards[idx]) for idx in set(requested)),
                        sum(mxfp4_state_nbytes(packed_shards[idx]) for idx in packed_shards) * 2)


if __name__ == "__main__":
    unittest.main()
