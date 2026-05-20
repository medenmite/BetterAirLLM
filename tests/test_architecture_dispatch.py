import json
import sys
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("transformers") is None:
    transformers_stub = ModuleType("transformers")
    transformers_stub.AutoConfig = SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)
    sys.modules.setdefault("transformers", transformers_stub)
SPEC = importlib.util.spec_from_file_location(
    "airllm_auto_model_for_tests",
    ROOT / "air_llm" / "airllm" / "auto_model.py",
)
auto_model_module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(auto_model_module)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "model_configs"


def _namespace_from_dict(value):
    if isinstance(value, dict):
        if "quant_method" in value:
            return value
        return SimpleNamespace(**{key: _namespace_from_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace_from_dict(item) for item in value]
    return value


class ArchitectureDispatchTests(unittest.TestCase):
    def setUp(self):
        self.original_from_pretrained = auto_model_module.AutoConfig.from_pretrained

        def fake_from_pretrained(name, *args, **kwargs):
            with (FIXTURES / f"{name}.json").open("r", encoding="utf-8") as handle:
                return _namespace_from_dict(json.load(handle))

        auto_model_module.AutoConfig.from_pretrained = fake_from_pretrained

    def tearDown(self):
        auto_model_module.AutoConfig.from_pretrained = self.original_from_pretrained

    def test_dispatches_supported_architecture_families(self):
        cases = {
            "llama": "BetterAirLLMLlama2",
            "mistral": "BetterAirLLMMistral",
            "qwen": "BetterAirLLMQWen",
            "qwen2": "BetterAirLLMQWen2",
            "qwen35_moe": "BetterAirLLMMoE",
            "gpt_oss_mxfp4": "BetterAirLLMMoE",
            "mixtral": "BetterAirLLMMoE",
            "chatglm": "BetterAirLLMChatGLM",
            "baichuan": "BetterAirLLMBaichuan",
            "internlm": "BetterAirLLMInternLM",
        }

        for fixture_name, expected_class in cases.items():
            with self.subTest(fixture=fixture_name):
                module_name, class_name = auto_model_module.AutoModel.get_module_class(fixture_name)
                self.assertEqual(module_name, "airllm")
                self.assertEqual(class_name, expected_class)

    def test_selects_gpt_oss_mxfp4_adapter_from_config(self):
        config = auto_model_module.AutoConfig.from_pretrained("gpt_oss_mxfp4")

        self.assertEqual(
            auto_model_module._selective_fused_adapter_for_config(config),
            "gpt_oss_mxfp4_reference",
        )

    def test_selects_qwen35_adapter_from_nested_text_config(self):
        config = auto_model_module.AutoConfig.from_pretrained("qwen35_moe")

        self.assertEqual(
            auto_model_module._selective_fused_adapter_for_config(config),
            "qwen3_5_moe",
        )


if __name__ == "__main__":
    unittest.main()
