
from transformers import GenerationConfig


from .betterairllm_base import BetterAirLLMBaseModel



class BetterAirLLMQWen2(BetterAirLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(BetterAirLLMQWen2, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False


