
from transformers import GenerationConfig

from .betterairllm_base import BetterAirLLMBaseModel



class BetterAirLLMMistral(BetterAirLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(BetterAirLLMMistral, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_generation_config(self):
        return GenerationConfig()


