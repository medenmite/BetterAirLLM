

from .airllm_base import BetterAirLLMBaseModel



class BetterAirLLMLlama2(BetterAirLLMBaseModel):
    def __init__(self, *args, **kwargs):
        super(BetterAirLLMLlama2, self).__init__(*args, **kwargs)

