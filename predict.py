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
    entity_types = ckpt['entity_types']
    relation_types = ckpt['relation_types']
    max_entity_num = ckpt['max_entity_num']
    max_relation_num = ckpt['max_relation_num']
    self.model = NERE(len(entity_types), len(relation_types), max_entity_num = max_entity_num, max_relation_num = max_relation_num, rel_weight_mode = rel_weight_mode)
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
    input_ids, attention_mask = inputs['input_ids'].to(device(self.device)), inputs['attention_mask'].to(device(self.device))
    entity_start, entity_stop, entity_tag, relation_head, relation_tail, relation_tag = self.model(input_ids, attention_mask)
    entity_start = np.argmax(entity_start.detach().cpu().numpy()[0], axis = -1)
    entity_stop = np.argmax(entity_stop.detach().cpu().numpy()[0], axis = -1)
    entity_tag = np.argmax(entity_tag.detach().cpu().numpy()[0], axis = -1)

