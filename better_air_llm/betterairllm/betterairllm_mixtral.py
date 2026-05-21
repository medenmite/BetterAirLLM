from transformers import GenerationConfig

from .betterairllm_moe import BetterAirLLMMoE


class BetterAirLLMMixtral(BetterAirLLMMoE):


    def __init__(self, *args, **kwargs):


        super(BetterAirLLMMixtral, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()


