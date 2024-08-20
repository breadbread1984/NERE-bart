#!/usr/bin/python3

from datasets import load_dataset, Split
import numpy as np

def load_conll04(tokenizer, max_length = 1024):
  trainset = load_dataset('DFKI-SLT/conll04', split = 'train', trust_remote_code = True)
  valset = load_dataset('DFKI-SLT/conll04', split = 'validation', trust_remote_code = True)
  max_entity_num = np.max([len(sample['entities']) for sample in trainset])
  max_relation_num = np.max([len(sample['relations']) for sample in trainset])
  entity_types = ['Peop', 'Loc', 'Org', 'Other']
  relation_types = ['Located_In', 'Work_For', 'OrgBased_In', 'Live_In', 'Kill']
  def preprocess(sample):
    inputs = tokenizer([sample['tokens']], is_split_into_words = True, return_tensors = 'np', padding = "max_length", max_length = max_length)
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]
    tokens_per_word = np.array([len(tokenizer.tokenize(word)) for word in sample['tokens']])
    entity_starts = list()
    entity_stops = list()
    entity_tags = list()
    for entity in sample['entities']:
      entity_starts.append(np.sum(tokens_per_word[:entity['start']]) + 1)
      entity_stops.append(np.sum(tokens_per_word[:entity['end']]) + 1)
      entity_tags.append(entity_types.index(entity['type']))
    entity_starts = np.pad(np.stack(entity_starts, axis = 0), (0, max_entity_num - len(entity_starts)), constant_values = 0)
    entity_stops = np.pad(np.stack(entity_stops, axis = 0), (0, max_entity_num - len(entity_stops)), constant_values = 0)
    entity_tags = np.pad(np.stack(entity_tags, axis = 0), (0, max_entity_num - len(entity_tags)), constant_values = len(entity_types))
    relation_heads = list()
    relation_tails = list()
    relation_tags = list()
    for relation in sample['relations']:
      relation_heads.append(relation['head'])
      relation_tails.append(relation['tail'])
      relation_tags.append(relation_types.index(relation['type']))
    relation_heads = np.pad(np.stack(relation_heads, axis = 0), (0, max_relation_num - len(relation_heads)), constant_values = 0)
    relation_tails = np.pad(np.stack(relation_tails, axis = 0), (0, max_relation_num - len(relation_tails)), constant_values = 0)
    relation_tags = np.pad(np.stack(relation_tags, axis = 0), (0, max_relation_num - len(relation_tags)), constant_values = len(relation_types))
    return {'words': sample['tokens'], 'input_ids': input_ids, 'attention_mask': attention_mask, 'entity_starts': entity_starts, 'entity_stops': entity_stops, 'entity_tags': entity_tags,
            'relation_heads': relation_heads, 'relation_tails': relation_tails, 'relation_tags': relation_tags}
  trainset = trainset.map(preprocess)
  trainset.set_format(type = 'torch', columns = ['words', 'input_ids', 'attention_mask', 'entity_starts', 'entity_stops', 'entity_tags', 'relation_heads', 'relation_tails', 'relation_tags'])
  valset = valset.map(preprocess)
  valset.set_format(type = 'torch', columns = ['words', 'input_ids', 'attention_mask', 'entity_starts', 'entity_stops', 'entity_tags', 'relation_heads', 'relation_tails', 'relation_tags'])
  return trainset, valset, {'entity_types': entity_types,
                            'relation_types': relation_types,
                            'max_entity_num': max_entity_num,
                            'max_relation_num': max_relation_num}

def load_docred():
  trainset = load_dataset('thunlp/docred', split = 'train_annotated', trust_remote_code = True)
  valset = load_dataset('thunlp/docred', split = 'validation', trust_remote_code = True)
  return trainset, valset

if __name__ == "__main__":
  from transformers import AutoTokenizer
  from torch.utils.data import DataLoader
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', add_prefix_space = True)
  trainset, valset, meta = load_conll04(tokenizer)
  train_loader = DataLoader(trainset, batch_size = 2)
  for sample in train_loader:
    print(sample['input_ids'].shape, sample['attention_mask'])
    print(sample['entity_starts'], type(sample['entity_starts']), sample['entity_starts'].shape)
    break
