from kangaroo.adapter import AdapterModel

class Lora(AdapterModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }