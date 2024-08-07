#!/usr/bin/python3

import torch
from torch import nn, device
from transformers import BartTokenizer, BartModel

class NERE(nn.Module):
  def __init__(self,):
    super(NERE, self).__init__()
    self.tokenizer = tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    self.backbone = BartModel.from_pretrained('facebook/bart-base')
  def forward(self, x):
    inputs = self.tokenizer(x, return_tensors = 'pt')
    inputs = inputs.to(self.device)
    outputs = self.backbone(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states

if __name__ == "__main__":
  model = NERE().to(device('cuda'))
  hidden = model("Hello, my dog is cute")
  print(hidden)

