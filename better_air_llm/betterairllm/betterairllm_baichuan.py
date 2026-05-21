from transformers import GenerationConfig

from .tokenization_baichuan import BaichuanTokenizer

from .betterairllm_base import BetterAirLLMBaseModel


class BetterAirLLMBaichuan(BetterAirLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(BetterAirLLMBaichuan, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_tokenizer(self, hf_token=None):

        return BaichuanTokenizer.from_pretrained(self.model_local_path, use_fast=False, trust_remote_code=True)

    def get_generation_config(self):
        return GenerationConfig()


