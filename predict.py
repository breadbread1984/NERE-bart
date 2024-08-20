#!/usr/bin/python3

import torch
from torch import load, device
from transformers import AutoTokenizer
import numpy as np
from models import NERE

class Predictor(object):
  def __init__(self, ckpt, dev = 'cuda'):
    assert dev in {'cuda', 'cpu'}
    ckpt = load(ckpt, map_location = device(dev))
    state_dict = ckpt['state_dict']
    state_dict = {(key.replace('module.','') if key.startswith('module.') else key): value for key,value in state_dict.items()}
    optimizer = ckpt['optimizer']
    scheduler = ckpt['scheduler']
    rel_weight_mode = ckpt['rel_weight_mode']
    self.entity_types = ckpt['entity_types']
    self.relation_types = ckpt['relation_types']
    self.max_entity_num = ckpt['max_entity_num']
    self.max_relation_num = ckpt['max_relation_num']
    self.model = NERE(len(self.entity_types), len(self.relation_types), max_entity_num = self.max_entity_num, max_relation_num = self.max_relation_num, rel_weight_mode = rel_weight_mode).to(device(dev))
    self.model.load_state_dict(state_dict)
    self.model.eval()
    self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', add_prefix_space = True)
    self.device = dev
  def call(self, input_ids, attention_mask):
    if 1 == len(input_ids.shape):
      input_ids = torch.unsqueeze(input_ids, dim = 0)
      attention_mask = torch.unsqueeze(attention_mask, dim = 0)
    input_ids, attention_mask = input_ids.to(device(self.device)), attention_mask.to(device(self.device))
    entity_start, entity_stop, entity_tag, relation_head, relation_tail, relation_tag = self.model(input_ids, attention_mask)
    # 1) entities
    entity_start = np.argmax(entity_start.detach().cpu().numpy()[0], axis = -1) # entity_start.shape = (max_entity_num,)
    entity_stop = np.argmax(entity_stop.detach().cpu().numpy()[0], axis = -1) # entity_stop.shape = (max_entity_num,)
    entity_tag = np.argmax(entity_tag.detach().cpu().numpy()[0], axis = -1) # entity_tag.shape = (max_entity_num,)
    entity_start = entity_start[entity_tag != len(self.entity_types)] # entity_start.shape = (entity_num,)
    entity_stop = entity_stop[entity_tag != len(self.entity_types)] # entity_stop.shape = (entity_num,)
    entity_tag = entity_tag[entity_tag != len(self.entity_types)] # entity_tag.shape = (entity_num,)
    # 2) relations
    relation_head = np.argmax(relation_head.detach().cpu().numpy()[0], axis = -1) # relation_head.shape = (max_relation_num,)
    relation_tail = np.argmax(relation_tail.detach().cpu().numpy()[0], axis = -1) # relation_head.shape = (max_relation_num,)
    relation_tag = np.argmax(relation_tag.detach().cpu().numpy()[0], axis = -1) # relation_tag.shape = (max_relation_num,)
    relation_head = relation_head[relation_tag != len(self.relation_types)] # relation_head.shape = (relation_num,)
    relation_tail = relation_tail[relation_tag != len(self.relation_types)] # relation_tail.shape = (relation_num,)
    relation_tag = relation_tag[relation_tag != len(self.relation_types)] # relation_tag.shape = (relation_num,)
    entities = [(s,e,t) for s,e,t in zip(entity_start, entity_stop, entity_tag)]
    relations = [(h,t,c) for h,t,c in zip(relation_head, relation_tail, relation_tag) if 0 <= h < len(entities) and 0 <= t < len(entities) and h != t]
    return entities, relations
  def __call__(self, text_or_words):
    if type(text_or_words) is str:
      text = text_or_words
      inputs = self.tokenizer(text, return_tensors = 'pt')
    elif type(text_or_words) is list:
      words = text_or_words
      inputs = self.tokenizer([words], is_split_into_words = True, return_tensors = 'pt')
    else:
      raise Exception('unknown input type!')
    return self.call(inputs['input_ids'], inputs['attention_mask'])
  def to_json(self, text_or_words, entities, relations):
    if type(text_or_words) is str:
      text = text_or_words
      inputs = self.tokenizer(text, return_tensors = 'np')
    elif type(text_or_words) is list:
      words = text_or_words
      text = ' '.join(words)
      inputs = self.tokenizer([words], is_split_into_words = True, return_tensors = 'np')
    tokens = inputs['input_ids'][0]
    results = {
      'original text': text,
      'tokens': tokens.tolist(),
      'entities': [
        {
          'entity': self.tokenizer.decode(tokens[entity[0]:entity[1]], skip_special_tokens = True),
          'start': entity[0],
          'stop': entity[1],
          'tag': self.entity_types[entity[2]],
        } 
        for entity in entities
      ],
      'relations': [
        {
          'head': self.tokenizer.decode(tokens[entities[relation[0]][0]:entities[relation[0]][1]], skip_special_tokens = True),
          'tail': self.tokenizer.decode(tokens[entities[relation[1]][0]:entities[relation[1]][1]], skip_special_tokens = True),
          'tag': self.relation_types[relation[2]]
        }
        for relation in relations
      ]
    }
    return results

if __name__ == "__main__":
  pred = Predictor('ckpt/model-ep220.pth', 'cuda')
  words = ['Newspaper', '`', 'Explains', "'", 'U.S.', 'Interests', 'Section', 'Events', 'FL1402001894', 'Havana', 'Radio', 'Reloj', 'Network', 'in', 'Spanish', '2100', 'GMT', '13', 'Feb', '94']
  entities, relations = pred(words)
  json = pred.to_json(words, entities, relations)
  print(json)
