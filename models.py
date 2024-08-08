#!/usr/bin/python3

import torch
from torch import nn, device
from transformers import AutoTokenizer, BartModel

class NERE(nn.Module):
  def __init__(self, triplets_num = 3):
    super(NERE, self).__init__()
    self.triplets_num = triplets_num
    self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    self.backbone = BartModel.from_pretrained('facebook/bart-base')
    self.embed = nn.Embedding(num_embeddings = self.backbone.config['max_position_embeddings'], embedding_dim = self.backbone.config['d_model'])
  def forward(self, x):
    encoder_inputs = self.tokenizer(x, return_tensors = 'pt')
    encoder_inputs = encoder_inputs.to(self.backbone.device) # encoder_inputs.shape = (1, length, d_model)
    decoder_inputs = torch.randint(0,self.backbone.config['max_position_embeddings'],size = (1,self.triplets_num))
    decoder_inputs = self.embed(decoder_inputs) # decoder_inputs.shape = (1, triplets_num, d_model)
    outputs = self.backbone(**encoder_inputs, decoder_inputs_embeds = decoder_inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states

if __name__ == "__main__":
  model = NERE().to(device('cuda'))
  hidden = model("Hello, my dog is cute")
  print(hidden.shape)
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
  entity_tokens = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str('Hello, my dog is cute')
  print(entity_tokens)
