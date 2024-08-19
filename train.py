#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists, splitext
import torch
from torch import nn, device, save, load, no_grad, any, isnan, autograd, sinh, log
from torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from create_datasets import load_conll04
from models import NERE

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', default = 'conll04', enum_values = {'conll04'}, help = 'available datasets')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_float('lr', default = 5.e-5, help = 'learning rate')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 50, help = 'number of epochs')
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
  torch.cuda_set_device(dist.get_rank())
  trainset_sampler = distributed.DistributedSampler(trainset)
  evalset_sampler = distributed.DistributedSampler(evalset)
  if dist.get_rank() == 0:
    print('trainset size: %d, evalset size: %d' % (len(trainset), len(evalset)))
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers, sampler = trainset_sampler, pin_memory = False)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers, sampler = evalset_sampler, pin_memory = False)
  model = NERE(meta['entity_types'], meta['relation_types'], max_entity_num = meta['max_entity_num'], max_relation_num = meta['max_relation_num'])
  model.to(device(FLAGS.device))
  model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
  criterion = nn.CrossEntropyLoss()
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
    for step, sample in enumerate(train_data_loader):
      optimizer.zero_grad()
      input_ids = sample['input_ids'].to(device(FLAGS.device))
      attention_mask = sample['attention_mask'].to(device(FLAGS.device))
      entity_starts = sample['entity_starts'].to(device(FLAGS.device))
      entity_stops = sample['entity_stops'].to(device(FLAGS.device))
      entity_tags = sample['entity_tags'].to(device(FlAGS.device))
      relation_heads = sample['relation_heads'].to(device(FLAGS.device))
      relation_tails = sample['relation_tails'].to(device(FLAGS.device))
      relation_tags = sample['relation_tags'].to(device(FLAGS.device))
      pred_entity_start, pred_entity_end, pred_entity_tag, pred_relation_head, pred_relation_tail, pred_tag = model(input_ids, attention_mask)
      loss1 = criterion(pred_entity_starts, entity_starts)
      loss2 = criterion(pred_entity_ends, entity_ends)
      loss3 = criterion(pred_entity_tags, entity_tags)
      loss4 = criterion(pred_relation_heads, relation_heads)
      loss5 = criterion(pred_relation_tails, relation_tails)
      loss6 = criterion(pred_relation_tags, relation_tags)
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
      eval_dataloader.sampler.set_epoch(epoch)
      model.eval()
      for sample in eval_dataloader:
        input_ids = sample['input_ids'].to(device(FLAGS.device))
        attention_mask = sample['attention_mask'].to(device(FLAGS.device))
        entity_starts = sample['entity_starts'].to(device(FLAGS.device))
        entity_stops = sample['entity_stops'].to(device(FLAGS.device))
        entity_tags = sample['entity_tags'].to(device(FlAGS.device))
        relation_heads = sample['relation_heads'].to(device(FLAGS.device))
        relation_tails = sample['relation_tails'].to(device(FLAGS.device))
        relation_tags = sample['relation_tags'].to(device(FLAGS.device))

