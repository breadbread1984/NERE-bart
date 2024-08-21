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
from models import NERE
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
  flags.DEFINE_enum('rel_weight_mode', default = 'independent', enum_values = {'independent','shared','fixed'}, help = 'how the relation decoder\'s weight are maintained')

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
  model = NERE(len(meta['entity_types']), len(meta['relation_types']), max_entity_num = meta['max_entity_num'], max_relation_num = meta['max_relation_num'], rel_weight_mode = FLAGS.rel_weight_mode)
  model.to(device(FLAGS.device))
  model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
  criterion = nn.CrossEntropyLoss().to(device(FLAGS.device))
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
      loss1 = criterion(pred_entity_starts.transpose(1,-1), entity_starts)
      loss2 = criterion(pred_entity_stops.transpose(1,-1), entity_stops)
      loss3 = criterion(pred_entity_tags.transpose(1,-1), entity_tags)
      loss4 = criterion(pred_relation_heads.transpose(1,-1), relation_heads)
      loss5 = criterion(pred_relation_tails.transpose(1,-1), relation_tails)
      loss6 = criterion(pred_relation_tags.transpose(1,-1), relation_tags)
      loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(train_dataloader) + step
      if global_steps % 100 == 0 and dist.get_rank() == 0:
        print('Step #%d Epoch #%d: loss %f, lr %f' % (global_steps, epoch, loss, scheduler.get_last_lr()[0]))
        tb_writer.add_scalar('entity start loss', loss1, global_steps)
        tb_writer.add_scalar('entity end loss', loss2, global_steps)
        tb_writer.add_scalar('entity tag loss', loss3, global_steps)
        tb_writer.add_scalar('relation head loss', loss4, global_steps)
        tb_writer.add_scalar('relation tail loss', loss5, global_steps)
        tb_writer.add_scalar('relation tag loss', loss6, global_steps)
    scheduler.step()
    if dist.get_rank() == 0:
      ckpt = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler,
        'rel_weight_mode': FLAGS.rel_weight_mode,
        **meta
      }
      save(ckpt, join(FLAGS.ckpt, 'model-ep%d.pth' % epoch))
      save(ckpt, join(FLAGS.ckpt, 'model.pth'))
      predictor = Predictor(join(FLAGS.ckpt, 'model.pth'), dev = FLAGS.device)
      pred_entities = list()
      label_entities = list()
      pred_relations = list()
      label_relations = list()
      for sample in evalset:
        entity_preds, relation_preds = predictor.call(sample['input_ids'], sample['attention_mask'])
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
      ent_prec, ent_rec, ent_f1 = get_metrics(pred_entities, label_entities)
      rel_prec, rel_rec, rel_f1 = get_metrics(pred_relations, label_relations)
      tb_writer.add_scalar('entity_precision', ent_prec, global_steps)
      tb_writer.add_scalar('entity_recall', ent_rec, global_steps)
      tb_writer.add_scalar('entity_f1', ent_f1, global_steps)
      tb_writer.add_scalar('relation_precision', rel_prec, global_steps)
      tb_writer.add_scalar('relation_recall', rel_rec, global_steps)
      tb_writer.add_scalar('relation_f1', rec_f1, global_steps)

if __name__ == "__main__":
  add_options()
  app.run(main)

