from adapter import AdapterModel

class LoraModel(AdapterModel):
    def __init__(self, config):
        super().__init__(config)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }
