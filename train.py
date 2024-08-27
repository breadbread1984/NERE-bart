#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import join, exists, splitext
import torch
from torch import nn, device, save, load, no_grad, any, isnan, autograd, sinh, log
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from create_datasets import load_conll04
from models import NERE, EntityCriterion, RelationCriterion
from predict import Predictor
from evaluation import get_metrics

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', default = 'conll04', enum_values = {'conll04'}, help = 'available datasets')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_float('lr', default = 5.e-5, help = 'learning rate')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 300, help = 'number of epochs')
  flags.DEFINE_integer('workers', default = 16, help = 'number of workers')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device')

def main(unused_argv):
  load_dataset = {
    'conll04': load_conll04,
  }[FLAGS.dataset]
  autograd.set_detect_anomaly(True)
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', add_prefix_space = True)
  trainset, evalset, meta = load_dataset(tokenizer)
  dist.init_process_group(backend = 'nccl')
  torch.cuda.set_device(dist.get_rank())
  trainset_sampler = distributed.DistributedSampler(trainset)
  if dist.get_rank() == 0:
    print('trainset size: %d, evalset size: %d' % (len(trainset), len(evalset)))
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers, sampler = trainset_sampler, pin_memory = False)
  model = NERE(len(meta['entity_types']), len(meta['relation_types']), max_entity_num = meta['max_entity_num'], max_relation_num = meta['max_relation_num'])
  model.to(device(FLAGS.device))
  model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
  ent_criterion = EntityCriterion(meta['entity_types']).to(device(FLAGS.device))
  rel_criterion = RelationCriterion(meta['relation_types']).to(device(FLAGS.device))
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  if dist.get_rank() == 0:
    if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
    tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  start_epoch = 0
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epochs):
    train_dataloader.sampler.set_epoch(epoch)
    model.train()
    for step, sample in enumerate(train_dataloader):
      optimizer.zero_grad()
      input_ids = sample['input_ids'].to(device(FLAGS.device))
      attention_mask = sample['attention_mask'].to(device(FLAGS.device))
      entity_starts = sample['entity_starts'].to(device(FLAGS.device))
      entity_stops = sample['entity_stops'].to(device(FLAGS.device))
      entity_tags = sample['entity_tags'].to(device(FLAGS.device))
      relation_heads = sample['relation_heads'].to(device(FLAGS.device))
      relation_tails = sample['relation_tails'].to(device(FLAGS.device))
      relation_tags = sample['relation_tags'].to(device(FLAGS.device))
      pred_entity_starts, pred_entity_stops, pred_entity_tags, pred_relation_heads, pred_relation_tails, pred_relation_tags = model(input_ids, attention_mask)
      entity_loss, indices = ent_criterion(pred_entity_starts, pred_entity_stops, pred_entity_tags, entity_starts, entity_stops, entity_tags)
      relation_loss = rel_criterion(indices, pred_relation_heads, pred_relation_tails, pred_relation_tags, relation_heads, relation_tails, relation_tags)
      loss = entity_loss + relation_loss
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(train_dataloader) + step
      if global_steps % 100 == 0 and dist.get_rank() == 0:
        print('Step #%d Epoch #%d: loss %f, lr %f' % (global_steps, epoch, loss, scheduler.get_last_lr()[0]))
        tb_writer.add_scalar('entity loss', entity_loss, global_steps)
        tb_writer.add_scalar('relation loss', relation_loss, global_steps)
    scheduler.step()
    if dist.get_rank() == 0:
      ckpt = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler,
        **meta
      }
      save(ckpt, join(FLAGS.ckpt, 'model.pth'))
      predictor = Predictor(join(FLAGS.ckpt, 'model.pth'), dev = FLAGS.device)
      pred_entities = list()
      label_entities = list()
      pred_relations = list()
      label_relations = list()
      for sample in evalset:
        entity_preds, relation_preds = predictor.call(sample['input_ids'], sample['attention_mask'])
        entity_idx = [(old_idx, b) for old_idx, (b,e,t) in enumerate(entity_preds)]
        entity_idx = list(sorted(entity_idx, key = lambda x: x[1])) # sort with entity begin token pos in ascent order
        entity_idx_map = {old_idx:new_idx for new_idx, (old_idx, b) in enumerate(entity_idx)}
        relation_preds = [(entity_idx_map[h], entity_idx_map[t], c) for h,t,c in relation_preds]
        entity_starts, entity_stops, entity_tags = sample['entity_starts'], sample['entity_stops'], sample['entity_tags']
        relation_heads, relation_tails, relation_tags = sample['relation_heads'], sample['relation_tails'], sample['relation_tags']
        entity_starts = entity_starts[entity_tags != len(meta['entity_types'])]
        entity_stops = entity_stops[entity_tags != len(meta['entity_types'])]
        entity_tags = entity_tags[entity_tags != len(meta['entity_types'])]
        relation_heads = relation_heads[relation_tags != len(meta['relation_types'])]
        relation_tails = relation_tails[relation_tags != len(meta['relation_types'])]
        relation_tags = relation_tags[relation_tags != len(meta['relation_types'])]
        entity_labels = [(b,e,t) for b,e,t in zip(entity_starts.cpu().numpy().tolist(), entity_stops.cpu().numpy().tolist(), entity_tags.cpu().numpy().tolist())]
        relation_labels = [(h,t,c) for h,t,c in zip(relation_heads.cpu().numpy().tolist(), relation_tails.cpu().numpy().tolist(), relation_tags.cpu().numpy().tolist())]
        pred_entities.append(entity_preds)
        label_entities.append(entity_labels)
        pred_relations.append(relation_preds)
        label_relations.append(relation_labels)
      ent_prec, ent_rec, ent_micro_f1, ent_macro_f1 = get_metrics(pred_entities, label_entities, len(meta['entity_types']))
      rel_prec, rel_rec, rel_micro_f1, rel_macro_f1 = get_metrics(pred_relations, label_relations, len(meta['relation_types']))
      tb_writer.add_scalar('entity_precision', ent_prec, global_steps)
      tb_writer.add_scalar('entity_recall', ent_rec, global_steps)
      tb_writer.add_scalar('entity_micro_f1', ent_micro_f1, global_steps)
      tb_writer.add_scalar('entity_macro_f1', ent_macro_f1, global_steps)
      tb_writer.add_scalar('relation_precision', rel_prec, global_steps)
      tb_writer.add_scalar('relation_recall', rel_rec, global_steps)
      tb_writer.add_scalar('relation_micro_f1', rel_micro_f1, global_steps)
      tb_writer.add_scalar('relation_macro_f1', rel_macro_f1, global_steps)

if __name__ == "__main__":
  add_options()
  app.run(main)

