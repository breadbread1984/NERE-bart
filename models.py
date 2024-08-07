#!/usr/bin/python3

import torch
from torch import nn, device
from transformers import BartTokenizer, BartModel

class NERE(nn.Module):
  def __init__(self,):
    super(NERE, self).__init__()
    self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    self.backbone = BartModel.from_pretrained('facebook/bart-base')
  def forward(self, x):
    inputs = self.tokenizer(x, return_tensors = 'pt')
    inputs = inputs.to(self.backbone.device)
    outputs = self.backbone(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states

if __name__ == "__main__":
  model = NERE().to(device('cuda'))
  hidden = model("Hello, my dog is cute")
  print(hidden.shape)
  tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
  entity_tokens = tokenizer.pre_tokenizer.pre_tokenize_str('Hello, my dog is cute')
  print(entity_tokens)
