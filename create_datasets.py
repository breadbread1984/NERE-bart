#!/usr/bin/python3

from datasets import load_dataset, Split
import numpy as np

def find_sublist_positions(lst, sublist):
    positions = []
    sublist_len = len(sublist)
    
    for i in range(len(lst) - sublist_len + 1):
        if lst[i:i + sublist_len] == sublist:
            positions.append(i)
    
    return positions

def load_conll04(tokenizer):
  trainset = load_dataset('DFKI-SLT/conll04', split = 'train', trust_remote_code = True)
  valset = load_dataset('DFKI-SLT/conll04', split = 'validation', trust_remote_code = True)
  max_entity_num = np.max([len(sample['entities']) for sample in trainset])
  max_relation_num = np.max([len(sample['relations']) for sample in trainset])
  entity_types = ['Peop', 'Loc', 'Org', 'Other']
  relation_types = ['Located_In', 'Work_For', 'OrgBased_In', 'Live_In', 'Kill']
  def preprocess(sample):
    tokens_per_word = np.array([len(tokenizer.tokenize(word)) for word in sample['tokens']])
    entity_starts = list()
    entity_stops = list()
    entity_tags = list()
    for entity in sample['entities']:
      entity_starts.append(np.sum(tokens_per_word[:entity['start']]) + 1)
      entity_stops.append(np.sum(tokens_per_word[:entity['end']]) + 1)
      entity_tags.append(entity_types.index(entity['type']))
    entity_starts = np.pad(np.concatenate(entity_starts, axis = 0), (0, max_entity_num - len(entity_starts)), constant_values = 0)
    entity_stops = np.pad(np.concatenate(entity_stops, axis = 0), (0, max_entity_num - len(entity_stops)), constant_values = 0)
    entity_tags = np.pad(np.concatenate(entity_tags, axis = 0), (0, max_entity_num - len(entity_tags)), constant_values = len(entity_types))
    relation_heads = list()
    relation_tails = list()
    relation_tags = list()
    for relation in sample['relations']:
      relation_heads.append(relation['head'])
      relation_tails.append(relation['tail'])
      relation_tags.append(relation_types.index(relation['type']))
    relation_heads = np.pad(np.concatenate(relation_heads, axis = 0), (0, max_relation_num - len(relation_heads)), constant_values = 0)
    relation_tails = np.pad(np.concatenate(relation_tails, axis = 0), (0, max_relation_num - len(relation_tails)), constant_values = 0)
    relation_tags = np.pad(np.concatenate(relation_tags, axis = 0), (0, max_relation_num - len(relation_tags)), constant_values = len(relation_types))
    return {'entity_starts': entity_starts, 'entity_stops': entity_stops, 'entity_tags': entity_tags,
            'relation_heads': relation_heads, 'relation_tails': relation_tails, 'relation_tags': relation_tags}
  return trainset.map(preprocess), valset.map(preprocess), max_entity_num, max_relation_num

def load_docred():
  trainset = load_dataset('thunlp/docred', split = 'train_annotated', trust_remote_code = True)
  valset = load_dataset('thunlp/docred', split = 'validation', trust_remote_code = True)
  return trainset, valset

if __name__ == "__main__":
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', add_prefix_space = True)
  trainset, valset, me, mr = load_conll04(tokenizer)
  for sample in trainset:
    print(sample)
    break
