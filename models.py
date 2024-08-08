#!/usr/bin/python3

import torch
from torch import nn, device
from torch.nn import functional as F
from transformers import AutoTokenizer, BartModel

class NERE(nn.Module):
  def __init__(self, label_num, triplets_num = 3):
    super(NERE, self).__init__()
    self.triplets_num = triplets_num
    self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    self.backbone = BartModel.from_pretrained('facebook/bart-base')
    self.embed = nn.Embedding(num_embeddings = self.backbone.config.max_position_embeddings, embedding_dim = self.backbone.config.d_model)
    self.head_entity_start = nn.Linear(self.backbone.config.d_model, self.backbone.config.max_position_embeddings)
    self.head_entity_end = nn.Linear(self.backbone.config.d_model, self.backbone.config.max_position_embeddings)
    self.head_entity_tag = nn.Linear(self.backbone.config.d_model, label_num)
    self.tail_entity_start = nn.Linear(self.backbone.config.d_model, self.backbone.config.max_position_embeddings)
    self.tail_entity_end = nn.Linear(self.backbone.config.d_model, self.backbone.config.max_position_embeddings)
    self.tail_entity_tag = nn.Linear(self.backbone.config.d_model, label_num)
  def forward(self, x):
    encoder_inputs = self.tokenizer(x, return_tensors = 'pt', padding = True)
    encoder_inputs = encoder_inputs.to(self.backbone.device) # encoder_inputs.shape = (batch, length, d_model)
    decoder_inputs = torch.randint(0, self.backbone.config.max_position_embeddings, size = (encoder_inputs['input_ids'].shape[0], self.triplets_num))
    decoder_inputs = decoder_inputs.to(self.backbone.device)
    decoder_inputs = self.embed(decoder_inputs) # decoder_inputs.shape = (batch, triplets_num, d_model)
    outputs = self.backbone(**encoder_inputs, decoder_inputs_embeds = decoder_inputs)
    last_hidden_states = outputs.last_hidden_state # last_hidden_state
    # head start
    head_start = self.head_entity_start(last_hidden_states)
    head_start = F.softmax(head_start, dim = -1) # entity_start.shape = (batch, triplets_num, max_position_embeddings)
    head_start = torch.argmax(head_start, dim = -1) # head_start.shape = (batch, triplets_num)
    # head end
    head_end = self.head_entity_end(last_hidden_states)
    head_end = F.softmax(head_end, dim = -1) # head_end.shape = (batch, triplets_num, max_position_embeddings)
    head_end = torch.argmax(head_end, dim = -1) # head_end.shape = (batch, triplets_num)
    # head tag
    head_tag = self.head_entity_tag(last_hidden_states)
    head_tag = F.softmax(head_tag, dim = -1) # head_tag.shape = (batch, triplets_num, label_num)
    head_tag = torch.argmax(head_tag, dim = -1) # head_tag.shape = (batch, triplets_num)
    # tail start
    tail_start = self.tail_entity_start(last_hidden_states)
    tail_start = F.softmax(tail_start, dim = -1)
    tail_start = torch.argmax(tail_start, dim = -1)
    # tail end
    tail_end = self.tail_entity_end(last_hidden_states)
    tail_end = F.softmax(tail_end, dim = -1)
    tail_end = torch.argmax(tail_end, dim = -1)
    # tail tag
    tail_tag = self.tail_entity_tag(last_hidden_states)
    tail_tag = F.softmax(tail_tag, dim = -1)
    tail_tag = torch.argmax(tail_tag, dim = -1)
    return head_start, head_end, head_tag, tail_start, tail_end, tail_tag

if __name__ == "__main__":
  model = NERE(label_num = 7).to(device('cuda'))
  hs,he,ht,ts,te,tt = model(["Hello, my dog is cute", "Hello the world!"])
  print(hs.shape,he.shape,ht.shape,ts.shape,te.shape,tt.shape)
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
  hidden = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str('Hello, my dog is cute')
  print(hidden)
