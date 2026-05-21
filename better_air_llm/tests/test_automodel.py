import sys
import unittest


from ..betterairllm.auto_model import AutoModel


class TestAutoModel(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': 'BetterAirLLMLlama2',
            'Qwen/Qwen-7B': 'BetterAirLLMQWen',
            'internlm/internlm-chat-7b': 'BetterAirLLMInternLM',
            'THUDM/chatglm3-6b-base': 'BetterAirLLMChatGLM',
            'baichuan-inc/Baichuan2-7B-Base': 'BetterAirLLMBaichuan',
            'mistralai/Mistral-7B-Instruct-v0.1': 'BetterAirLLMMistral',
            'mistralai/Mixtral-8x7B-v0.1': 'BetterAirLLMMoE'
        }


        for k,v in mapping_dict.items():
            module, cls = AutoModel.get_module_class(k)
            self.assertEqual(cls, v, f"expecting {v}")

