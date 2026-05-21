from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .betterairllm_llama_mlx import BetterAirLLMLlamaMlx
    from .auto_model import AutoModel
else:
    from .betterairllm import BetterAirLLMLlama2
    from .betterairllm_chatglm import BetterAirLLMChatGLM
    from .betterairllm_qwen import BetterAirLLMQWen
    from .betterairllm_qwen2 import BetterAirLLMQWen2
    try:
        from .betterairllm_baichuan import BetterAirLLMBaichuan
    except ImportError:
        BetterAirLLMBaichuan = None
    from .betterairllm_internlm import BetterAirLLMInternLM
    from .betterairllm_mistral import BetterAirLLMMistral
    from .betterairllm_moe import BetterAirLLMMoE
    from .betterairllm_mixtral import BetterAirLLMMixtral
    from .betterairllm_base import BetterAirLLMBaseModel
    from .selective_fused_moe import (
        SelectiveFusedMoEAdapter,
        FakeFusedMoEAdapter,
        GptOssSelectiveFusedMoEAdapter,
        Qwen35SelectiveFusedMoEAdapter,
    )
    from .gpt_oss_mxfp4 import (
        dequantize_mxfp4_expert,
        dequantize_mxfp4_projection,
        load_gpt_oss_expert_mxfp4_shard,
        run_gpt_oss_selected_expert_reference,
    )
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException

