#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists
import numpy as np
from transformers import AutoTokenizer
from predict import Predictor
from create_datasets import load_conll04

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', default = 'conll04', enum_values = {'conll04'}, help = 'available datasets')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device')

def get_metrics(preds_list, labels_list, type_num):
  # NOTE: copy code from https://github.com/LorrinWWW/two-are-better-than-one/blob/a75de25e436a02f58bc512de2f841d621be40daa/data/joint_data.py#L127
  # micro f1
  n_correct, n_pred, n_label = 0, 0, 0
  for preds, labels in zip(preds_list, labels_list):
    preds = set(preds)
    labels = set(labels)
    n_pred += len(preds)
    n_label += len(labels)
    n_correct += len(preds & labels)
  precision = n_correct / (n_pred + 1e-8) # TP / (TP + FP)
  recall = n_correct / (n_label + 1e-8) # TP / (TP + FN)
  micro_f1 = 2 / (1/(precision+1e-8) + 1/(recall+1e-8) + 1e-8)
  # macro f1
  f1 = list()
  for i in range(type_num):
    n_correct, n_pred, n_label = 0,0,0
    preds_list_i = list(filter(lambda x: x[2] == i, preds_list))
    labels_list_i = list(filter(lambda x: x[2] == i, labels_list))
    for preds, labels in zip(preds_list_i, labels_list_i):
      preds = set(preds)
      labels = set(labels)
      n_pred += len(preds)
      n_label += len(labels)
      n_correct += len(preds & labels)
    precision = n_correct / (n_pred + 1e-8) # TP_i / (TP_i + FP_i)
    recall = n_correct / (n_label + 1e-8) # TP_i / (TP_i + FN_i)
    f1.append(2 / (1/(precision+1e-8) + 1/(recall+1e-8) + 1e-8))
  macro_f1 = np.mean(f1)
  return precision, recall, micro_f1, macro_f1

def main(unused_argv):
  if not exists(join(FLAGS.ckpt, 'model.pth')):
    raise Exception('cannot find model.pth under directory designated by FLAGS.ckpt')
  load_dataset = {
    'conll04': load_conll04,
  }[FLAGS.dataset]
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', add_prefix_space = True)
  _, evalset, meta = load_dataset(tokenizer)
  entity_types = meta['entity_types']
  relation_types = meta['relation_types']
  predictor = Predictor(join(FLAGS.ckpt, 'model.pth'), dev = FLAGS.device)
  pred_entities = list()
  label_entities = list()
  pred_relations = list()
  label_relations = list()
  for sample in evalset:
    entity_preds, relation_preds = predictor.call(sample['input_ids'], sample['attention_mask'])
    entity_starts, entity_stops, entity_tags = sample['entity_starts'], sample['entity_stops'], sample['entity_tags']
    relation_heads, relation_tails, relation_tags = sample['relation_heads'], sample['relation_tails'], sample['relation_tags']
    entity_starts = entity_starts[entity_tags != len(entity_types)]
    entity_stops = entity_stops[entity_tags != len(entity_types)]
    entity_tags = entity_tags[entity_tags != len(entity_types)]
    relation_heads = relation_heads[relation_tags != len(relation_types)]
    relation_tails = relation_tails[relation_tags != len(relation_types)]
    relation_tags = relation_tags[relation_tags != len(relation_types)]
    entity_labels = [(b,e,t) for b,e,t in zip(entity_starts.cpu().numpy().tolist(), entity_stops.cpu().numpy().tolist(), entity_tags.cpu().numpy().tolist())]
    relation_labels = [(h,t,c) for h,t,c in zip(relation_heads.cpu().numpy().tolist(), relation_tails.cpu().numpy().tolist(), relation_tags.cpu().numpy().tolist())]
    pred_entities.append(entity_preds)
    label_entities.append(entity_labels)
    pred_relations.append(relation_preds)
    label_relations.append(relation_labels)
  ent_prec, ent_rec, ent_micro_f1, ent_macro_f1 = get_metrics(pred_entities, label_entities, len(entity_types))
  rel_prec, rel_rec, rel_micro_f1, rel_macro_f1 = get_metrics(pred_relations, label_relations, len(relation_types))
  print(f'entity| precision: {ent_prec} recall: {ent_rec} micro f1: {ent_micro_f1} macro f1: {ent_macro_f1}')
  print(f'relation| precision: {rel_prec} recall: {rel_rec} micro f1: {rel_micro_f1} macro f1: {rel_macro_f1}')

if __name__ == "__main__":
  add_options()
  app.run(main)
