import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from airllm import BetterAirLLMLlamaMlx


def _architecture_name(config):
    architectures = getattr(config, "architectures", None) or []
    return architectures[0] if architectures else ""


def _is_moe_config(config):
    architecture = _architecture_name(config).lower()
    model_type = getattr(config, "model_type", "").lower()

    moe_markers = (
        "moe",
        "mixtral",
        "deepseek",
        "dbrx",
        "olmoe",
        "phimoe",
        "gpt_oss",
        "gpt-oss",
    )
    if any(marker in architecture or marker in model_type for marker in moe_markers):
        return True

    expert_attrs = (
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_experts_per_tok",
        "num_selected_experts",
        "moe_intermediate_size",
    )
    return any(hasattr(config, attr) for attr in expert_attrs)


def _selective_fused_adapter_for_config(config):
    model_type = getattr(config, "model_type", "").lower()
    architecture = _architecture_name(config).lower()
    text_config = getattr(config, "text_config", None)
    text_model_type = getattr(text_config, "model_type", "").lower() if text_config is not None else ""
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} or text_model_type == "qwen3_5_moe_text":
        return "qwen3_5_moe"
    if model_type == "gpt_oss" or "gptoss" in architecture or "gpt_oss" in architecture:
        quantization = getattr(config, "quantization_config", None) or {}
        if isinstance(quantization, dict) and quantization.get("quant_method") == "mxfp4":
            return "gpt_oss_mxfp4_reference"
        return "gpt_oss"
    return None

class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )
    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            print(f"using hf_token")
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        architecture = _architecture_name(config)

        if _is_moe_config(config):
            return "airllm", "BetterAirLLMMoE"
        if "Qwen2ForCausalLM" in architecture:
            return "airllm", "BetterAirLLMQWen2"
        elif "QWen" in architecture:
            return "airllm", "BetterAirLLMQWen"
        elif "Baichuan" in architecture:
            return "airllm", "BetterAirLLMBaichuan"
        elif "ChatGLM" in architecture:
            return "airllm", "BetterAirLLMChatGLM"
        elif "InternLM" in architecture:
            return "airllm", "BetterAirLLMInternLM"
        elif "Mistral" in architecture:
            return "airllm", "BetterAirLLMMistral"
        elif "Llama" in architecture:
            return "airllm", "BetterAirLLMLlama2"
        else:
            print(f"unknown architecture: {architecture}, try to use Llama2...")
            return "airllm", "BetterAirLLMLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        if is_on_mac_os:
            return BetterAirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, ** kwargs)

        module, cls = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        config = None
        if cls == "BetterAirLLMMoE" and "moe_selective_fused_adapter_name" not in kwargs:
            if 'hf_token' in kwargs:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token'])
            else:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            adapter_name = _selective_fused_adapter_for_config(config)
            if adapter_name is not None:
                kwargs["moe_selective_fused_adapter_name"] = adapter_name
        if cls == "BetterAirLLMMoE":
            if config is None:
                if 'hf_token' in kwargs:
                    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token'])
                else:
                    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            model_type = getattr(config, "model_type", "").lower()
            text_config = getattr(config, "text_config", None)
            text_model_type = getattr(text_config, "model_type", "").lower() if text_config is not None else ""
            if model_type == "qwen3_5_moe" or text_model_type == "qwen3_5_moe_text":
                kwargs["qwen3_5_moe_layout"] = True
        module = importlib.import_module(module)
        class_ = getattr(module, cls)
        return class_(pretrained_model_name_or_path, *inputs, ** kwargs)
