from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .airllm_llama_mlx import BetterAirLLMLlamaMlx
    from .auto_model import AutoModel
else:
    from .airllm import BetterAirLLMLlama2
    from .airllm_chatglm import BetterAirLLMChatGLM
    from .airllm_qwen import BetterAirLLMQWen
    from .airllm_qwen2 import BetterAirLLMQWen2
    try:
        from .airllm_baichuan import BetterAirLLMBaichuan
    except ImportError:
        BetterAirLLMBaichuan = None
    from .airllm_internlm import BetterAirLLMInternLM
    from .airllm_mistral import BetterAirLLMMistral
    from .airllm_moe import BetterAirLLMMoE
    from .airllm_mixtral import BetterAirLLMMixtral
    from .airllm_base import BetterAirLLMBaseModel
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

