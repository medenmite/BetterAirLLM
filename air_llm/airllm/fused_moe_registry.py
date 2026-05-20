"""Known fused MoE layouts.

This registry is intentionally conservative. A family is marked
``selective_runtime=False`` until BetterAirLLM has a correctness test for that
exact router, activation, tensor orientation, and bias layout.
"""


FUSED_MOE_LAYOUTS = {
    "fake_fused_moe": {
        "adapter_name": "fake_fused_moe",
        "selective_runtime": True,
        "tensor_names": ("experts.gate_up_proj.weight", "experts.down_proj.weight"),
        "router": "linear -> topk -> softmax(topk)",
        "activation": "silu(gate) * up",
        "layout": {
            "gate_up_proj.weight": "[num_experts, 2 * intermediate_dim, hidden_dim]",
            "down_proj.weight": "[num_experts, hidden_dim, intermediate_dim]",
        },
        "correctness_test": "air_llm.tests.test_selective_fused_moe",
    },
    "mixtral": {
        "adapter_name": None,
        "selective_runtime": False,
        "tensor_names": ("experts.gate_up_proj", "experts.down_proj"),
        "router": "MixtralTopKRouter: linear -> softmax -> topk -> renormalize",
        "activation": "silu(gate) * up",
        "layout": {
            "gate_up_proj": "[num_experts, 2 * intermediate_dim, hidden_dim]",
            "down_proj": "[num_experts, hidden_dim, intermediate_dim]",
        },
        "correctness_test": None,
    },
    "qwen_moe": {
        "adapter_name": None,
        "selective_runtime": False,
        "tensor_names": ("experts.gate_up_proj", "experts.down_proj"),
        "router": "Qwen MoE TopKRouter: linear -> softmax -> topk, optional renormalize",
        "activation": "hidden_act(gate) * up",
        "layout": {
            "gate_up_proj": "[num_experts, 2 * intermediate_dim, hidden_dim]",
            "down_proj": "[num_experts, hidden_dim, intermediate_dim]",
        },
        "correctness_test": None,
    },
    "qwen3_5_moe": {
        "adapter_name": "qwen3_5_moe",
        "selective_runtime": True,
        "tensor_names": ("experts.gate_up_proj", "experts.down_proj"),
        "router": "Qwen3_5MoeTopKRouter: linear -> softmax -> topk -> renormalize",
        "activation": "silu(gate) * up plus dense shared_expert outside routed experts",
        "layout": {
            "gate_up_proj": "[num_experts, 2 * moe_intermediate_size, hidden_dim]",
            "down_proj": "[num_experts, hidden_dim, moe_intermediate_size]",
        },
        "correctness_test": "air_llm.tests.test_qwen35_selective_moe",
    },
    "deepseek_v3": {
        "adapter_name": None,
        "selective_runtime": False,
        "tensor_names": ("experts.gate_up_proj", "experts.down_proj"),
        "router": "sigmoid router with group-limited topk and correction bias",
        "activation": "silu(gate) * up plus shared experts outside routed experts",
        "layout": {
            "gate_up_proj": "[num_experts, 2 * intermediate_dim, hidden_dim]",
            "down_proj": "[num_experts, hidden_dim, intermediate_dim]",
        },
        "correctness_test": None,
    },
    "gpt_oss": {
        "adapter_name": "gpt_oss_mxfp4_reference",
        "selective_runtime": True,
        "tensor_names": (
            "experts.gate_up_proj_blocks",
            "experts.gate_up_proj_scales",
            "experts.gate_up_proj_bias",
            "experts.down_proj_blocks",
            "experts.down_proj_scales",
            "experts.down_proj_bias",
        ),
        "router": "GptOssTopKRouter: linear+bias -> topk logits -> softmax(topk)",
        "activation": "OpenAI gated clamp: gate/up interleaved, clamp, sigmoid alpha=1.702, (up+1)*glu",
        "mxfp4_execution": "reference_dequant",
        "layout": {
            "gate_up_proj_blocks": "[num_experts, 2 * intermediate_dim, hidden_dim // 32, 16]",
            "gate_up_proj_scales": "[num_experts, 2 * intermediate_dim, hidden_dim // 32]",
            "gate_up_proj_bias": "[num_experts, 2 * intermediate_dim]",
            "down_proj_blocks": "[num_experts, hidden_dim, intermediate_dim // 32, 16]",
            "down_proj_scales": "[num_experts, hidden_dim, intermediate_dim // 32]",
            "down_proj_bias": "[num_experts, hidden_dim]",
        },
        "correctness_test": "air_llm.tests.test_gpt_oss_mxfp4_reference",
    },
}


def get_fused_moe_layout(model_type):
    return FUSED_MOE_LAYOUTS.get((model_type or "").lower())
