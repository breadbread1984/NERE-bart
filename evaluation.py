#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists
from predict import Predictor
from create_datasets import load_conll04

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', default = 'conll04', enum_values = {'conll04'}, help = 'available datasets')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device')

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
    entity_tags = entity_tags[entity_tags != len(entity_tags)]
    entity_labels = [(b,e,t) for b,e,t in zip(entity_starts, entity_stops, entity_tags)]
    relation_labels = [(h,t,c) for h,t,c in zip(relation_heads, relation_tails, relation_tags)]
    pred_entities.append(entity_preds)
    label_entities.append(entity_labels)


