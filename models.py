#!/usr/bin/python3

import torch
from torch import nn, device
from torch.nn import functional as F
from transformers import AutoTokenizer, BartModel
from transformers.models.bart.modeling_bart import BartDecoder

class NERE(nn.Module):
  def __init__(self, entity_tag_num, relation_tag_num, max_entity_num = 10, max_relation_num = 10, triplets_num = 3):
    super(NERE, self).__init__()
    self.triplets_num = triplets_num
    self.max_entity_num = max_entity_num
    self.max_relation_num = max_relation_num
    self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    self.encoder_and_entity_decoder = BartModel.from_pretrained('facebook/bart-base')
    self.relation_decoder = BartDecoder.from_pretrained('facebook/bart-base')
    self.entity_embed = nn.Embedding(num_embeddings = max_entity_num, embedding_dim = self.encoder_and_entity_decoder.config.d_model)
    self.entity_start = nn.Linear(self.encoder_and_entity_decoder.config.d_model, self.encoder_and_entity_decoder.config.max_position_embeddings)
    self.entity_end = nn.Linear(self.encoder_and_entity_decoder.config.d_model, self.encoder_and_entity_decoder.config.max_position_embeddings)
    self.entity_tag = nn.Linear(self.encoder_and_entity_decoder.config.d_model, entity_tag_num + 1)
    self.relation_embed = nn.Embedding(num_embeddings = max_relation_num, embedding_dim = self.encoder_and_entity_decoder.config.d_model)
    self.relation_head = nn.Linear(self.encoder_and_entity_decoder.config.d_model, max_entity_num)
    self.relation_tail = nn.Linear(self.encoder_and_entity_decoder.config.d_model, max_entity_num)
    self.relation_tag = nn.Linear(self.encoder_and_entity_decoder.config.d_model, relation_tag_num + 1)
  def forward(self, input_ids, attention_mask):
    # 1) entity prediction
    entity_embed_inputs = torch.tile(torch.unsqueeze(torch.range(0, self.max_entity_num - 1, dtype = torch.int32), dim = 0), (input_ids.shape[0], 1)) # entity_embed_inputs.shape = (batch, max_entity_num)
    entity_embed_inputs = entity_embed_inputs.to(self.encoder_and_entity_decoder.device)
    entity_embed_inputs = self.entity_embed(entity_embed_inputs) # entity_embed_inputs.shape = (batch, max_entity_num, d_model)
    outputs = self.encoder_and_entity_decoder(input_ids = input_ids, attention_mask = attention_mask, decoder_inputs_embeds = entity_embed_inputs)
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
    entity_tag = F.softmax(entity_tag, dim = -1) # entity_tag.shape = (batch, triplets_num, entity_tag_num)
    entity_tag = torch.argmax(entity_tag, dim = -1) # entity_tag.shape = (batch, triplets_num)
    # 2) relationship prediction
    relation_embed_inputs = torch.tile(torch.unsqueeze(torch.range(0, self.max_relation_num - 1, dtype = torch.int32), dim = 0), (input_ids.shape[0], 1)) # relation_embed_inputs.shape = (batch, max_relation_num)
    relation_embed_inputs = relation_embed_inputs.to(self.encoder_and_entity_decoder.device)
    relation_embed_inputs = self.relation_embed(relation_embed_inputs) # relation_embed_inputs.shape = (batch, max_relation_num, d_model)
    outputs = self.relation_decoder(encoder_hidden_states = last_hidden_states, inputs_embeds = relation_embed_inputs)
    last_hidden_states = outputs.last_hidden_state
    # relation head
    relation_head = self.relation_head(last_hidden_states)
    relation_head = F.softmax(relation_head, dim = -1)
    relation_head = torch.argmax(relation_head, dim = -1)
    # relation tail
    relation_tail = self.relation_tail(last_hidden_states)
    relation_tail = F.softmax(relation_tail, dim = -1)
    relation_tail = torch.argmax(relation_tail, dim = -1)
    # relation tag
    relation_tag = self.relation_tag(last_hidden_states)
    relation_tag = F.softmax(relation_tag, dim = -1)
    relation_tag = torch.argmax(relation_tag, dim = -1)
    return entity_start, entity_end, entity_tag, relation_head, relation_tail, relation_tag

if __name__ == "__main__":
  d = 'cpu'
  model = NERE(entity_tag_num = 7, relation_tag_num = 5, ).to(device(d))
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
  inputs = tokenizer(["Hello, my dog is cute", "Hello the world!"], return_tensors = 'pt', padding = True)
  es,ee,et,rh,rt,rt = model(inputs['input_ids'].to(device(d)), inputs['attention_mask'].to(device(d)))
  print(es.shape,ee.shape,et.shape,rh.shape,rt.shape,rt.shape)
  hidden = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str('Hello, my dog is cute')
  print(hidden)
