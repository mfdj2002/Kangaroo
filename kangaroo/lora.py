import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

class Lora(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }