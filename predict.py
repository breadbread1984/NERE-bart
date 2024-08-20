#!/usr/bin/python3

from torch import load, device
from transformers import AutoTokenizer
from models import NERE

class Predictor(object):
  def __init__(self, ckpt, dev = 'cuda'):
    assert dev in {'cuda', 'cpu'}
    ckpt = load(ckpt, map_location = device(dev))
    state_dict = ckpt['state_dict']
    optimizer = ckpt['optimizer']
    scheduler = ckpt['scheduler']
    rel_weight_mode = ckpt['rel_weight_mode']
    self.entity_types = ckpt['entity_types']
    self.relation_types = ckpt['relation_types']
    self.max_entity_num = ckpt['max_entity_num']
    self.max_relation_num = ckpt['max_relation_num']
    self.model = NERE(len(self.entity_types), len(self.relation_types), max_entity_num = self.max_entity_num, max_relation_num = self.max_relation_num, rel_weight_mode = rel_weight_mode)
    self.model.eval()
    self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', add_prefix_space = True)
    self.device = dev
  def __call__(self, text_or_words):
    if type(text_or_words) is str:
      text = text_or_words
      inputs = self.tokenizer(text, return_tensors = 'pt', padding = 'max_length', max_length = 1024)
    elif type(text_or_words) is list:
      words = text_or_words
      inputs = self.tokenizer([words], is_split_into_words = True, return_tensors = 'pt', padding = 'max_length', max_length = 1024)
    else:
      raise Exception('unknown input type!')
    input_ids, attention_mask = inputs['input_ids'].to(device(self.device)), inputs['attention_mask'].to(device(self.device))
    entity_start, entity_stop, entity_tag, relation_head, relation_tail, relation_tag = self.model(input_ids, attention_mask)
    # 1) entities
    entity_start = np.argmax(entity_start.detach().cpu().numpy()[0], axis = -1) # entity_start.shape = (max_entity_num,)
    entity_stop = np.argmax(entity_stop.detach().cpu().numpy()[0], axis = -1) # entity_stop.shape = (max_entity_num,)
    entity_tag = np.argmax(entity_tag.detach().cpu().numpy()[0], axis = -1) # entity_tag.shape = (max_entity_num,)
    entity_start = np.boolean_mask(entity_tag != len(self.entity_types), entity_start) # entity_start.shape = (entity_num,)
    entity_stop = np.boolean_mask(entity_tag != len(self.entity_types), entity_stop) # entity_stop.shape = (entity_num,)
    entity_tag = np.boolean_mask(entity_tag != len(self.entity_types), entity_tag) # entity_tag.shape = (entity_num,)
    # 2) relations
    relation_head = np.argmax(relation_head.detach().cpu().numpy()[0], axis = -1) # relation_head.shape = (max_relation_num,)
    relation_tail = np.argmax(relation_tail.detach().cpu().numpy()[0], axis = -1) # relation_head.shape = (max_relation_num,)
    relation_tag = np.argmax(relation_tag.detach().cpu().numpy()[0], axis = -1) # relation_tag.shape = (max_relation_num,)
    relation_head = np.boolean_mask(relation_tag != len(self.relation_types), relation_head) # relation_head.shape = (relation_num,)
    relation_tail = np.boolean_mask(relation_tag != len(self.relation_types), relation_tail) # relation_tail.shape = (relation_num,)
    relation_tag = np.boolean_mask(relation_tag != len(self.relation_types), relation_tag) # relation_tag.shape = (relation_num,)

