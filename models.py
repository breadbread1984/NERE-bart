#!/usr/bin/python3

import torch
from torch import nn, device
from torch.nn import functional as F
from transformers import AutoTokenizer, BartModel
from transformers.models.bart.modeling_bart import BartDecoder

class NERE(nn.Module):
  def __init__(self, label_num, max_entity_num, max_relation_num, triplets_num = 3):
    super(NERE, self).__init__()
    self.triplets_num = triplets_num
    self.max_entity_num = max_entity_num
    self.max_relation_num = max_relation_num
    self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    self.backbone = BartModel.from_pretrained('facebook/bart-base')
    self.decoder = BartModelDecoder.from_pretrained('facebook/bart-base')
    self.entity_embed = nn.Embedding(num_embeddings = max_entity_num, embedding_dim = self.backbone.config.d_model)
    self.entity_start = nn.Linear(self.backbone.config.d_model, self.backbone.config.max_position_embeddings)
    self.entity_end = nn.Linear(self.backbone.config.d_model, self.backbone.config.max_position_embeddings)
    self.entity_tag = nn.Linear(self.backbone.config.d_model, label_num)
    self.relation_embed = nn.Embedding(num_embeddings = max_relation_num, embedding_dim = self.backbone.config.d_model)

  def forward(self, x):
    encoder_inputs = self.tokenizer(x, return_tensors = 'pt', padding = True)
    encoder_inputs = encoder_inputs.to(self.backbone.device) # encoder_inputs.shape = (batch, length, d_model)
    entity_embed_inputs = torch.tile(torch.unsqueeze(torch.range(0,self.max_entity_num - 1), dim = 0), (encoder_inputs['input_ids'].shape[0], 1)) # entity_embed_inputs.shape = (batch, max_entity_num)
    entity_embed_inputs = entity_embed_inputs.to(self.backbone.device)
    entity_embed_inputs = self.entity_embed(entity_embed_inputs) # entity_embed_inputs.shape = (batch, max_entity_num, d_model)
    outputs = self.backbone(**encoder_inputs, entity_embed_inputs_embeds = entity_embed_inputs)
    last_hidden_states = outputs.last_hidden_state # last_hidden_state
    # entity start
    entity_start = self.entity_start(last_hidden_states)
    entity_start = F.softmax(entity_start, dim = -1) # entity_start.shape = (batch, triplets_num, max_position_embeddings)
    entity_start = torch.argmax(entity_start, dim = -1) # entity_start.shape = (batch, triplets_num)
    # entity end
    entity_end = self.entity_end(last_hidden_states)
    entity_end = F.softmax(entity_end, dim = -1) # entity_end.shape = (batch, triplets_num, max_position_embeddings)
    entity_end = torch.argmax(entity_end, dim = -1) # entity_end.shape = (batch, triplets_num)
    # entity tag
    entity_tag = self.entity_tag(last_hidden_states)
    entity_tag = F.softmax(entity_tag, dim = -1) # entity_tag.shape = (batch, triplets_num, label_num)
    entity_tag = torch.argmax(entity_tag, dim = -1) # entity_tag.shape = (batch, triplets_num)
    return entity_start, entity_end, entity_tag

if __name__ == "__main__":
  model = NERE(label_num = 7).to(device('cuda'))
  es,ee,et = model(["Hello, my dog is cute", "Hello the world!"])
  print(es.shape,ee.shape,et.shape)
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
  hidden = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str('Hello, my dog is cute')
  print(hidden)
