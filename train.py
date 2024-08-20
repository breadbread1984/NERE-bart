#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import join, exists, splitext
import torch
from torch import nn, device, save, load, no_grad, any, isnan, autograd, sinh, log
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torcheval.metrics import MulticlassAccuracy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from create_datasets import load_conll04
from models import NERE

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', default = 'conll04', enum_values = {'conll04'}, help = 'available datasets')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_float('lr', default = 5.e-5, help = 'learning rate')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 300, help = 'number of epochs')
  flags.DEFINE_integer('workers', default = 16, help = 'number of workers')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device')
  flags.DEFINE_enum('decoder_weight_mode', default = 'independent', enum_values = {'independent','shared','fixed'}, help = 'how the decoder\'s weight are maintained')

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
  evalset_sampler = distributed.DistributedSampler(evalset)
  if dist.get_rank() == 0:
    print('trainset size: %d, evalset size: %d' % (len(trainset), len(evalset)))
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers, sampler = trainset_sampler, pin_memory = False)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers, sampler = evalset_sampler, pin_memory = False)
  model = NERE(len(meta['entity_types']), len(meta['relation_types']), max_entity_num = meta['max_entity_num'], max_relation_num = meta['max_relation_num'], rel_weight_mode = FLAGS.decoder_weight_mode)
  model.to(device(FLAGS.device))
  model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
  criterion = nn.CrossEntropyLoss().to(device(FLAGS.device))
  entity_start_accuracy = MulticlassAccuracy().to(device(FLAGS.device))
  entity_stop_accuracy = MulticlassAccuracy().to(device(FLAGS.device))
  entity_tag_accuracy = MulticlassAccuracy().to(device(FLAGS.device))
  relation_head_accuracy = MulticlassAccuracy().to(device(FLAGS.device))
  relation_tail_accuracy = MulticlassAccuracy().to(device(FLAGS.device))
  relation_tag_accuracy = MulticlassAccuracy().to(device(FLAGS.device))
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
      eval_dataloader.sampler.set_epoch(epoch)
      model.eval()
      for sample in eval_dataloader:
        input_ids = sample['input_ids'].to(device(FLAGS.device))
        attention_mask = sample['attention_mask'].to(device(FLAGS.device))
        entity_starts = sample['entity_starts'].to(device(FLAGS.device))
        entity_stops = sample['entity_stops'].to(device(FLAGS.device))
        entity_tags = sample['entity_tags'].to(device(FLAGS.device))
        relation_heads = sample['relation_heads'].to(device(FLAGS.device))
        relation_tails = sample['relation_tails'].to(device(FLAGS.device))
        relation_tags = sample['relation_tags'].to(device(FLAGS.device))
        pred_entity_starts, pred_entity_stops, pred_entity_tags, pred_relation_heads, pred_relation_tails, pred_relation_tags = model(input_ids, attention_mask)
        entity_start_accuracy.update(torch.flatten(pred_entity_starts, end_dim = -2), torch.flatten(entity_starts))
        entity_stop_accuracy.update(torch.flatten(pred_entity_stops, end_dim = -2), torch.flatten(entity_stops))
        entity_tag_accuracy.update(torch.flatten(pred_entity_tags, end_dim = -2), torch.flatten(entity_tags))
        relation_head_accuracy.update(torch.flatten(pred_relation_heads, end_dim = -2), torch.flatten(relation_heads))
        relation_tail_accuracy.update(torch.flatten(pred_relation_tails, end_dim = -2), torch.flatten(relation_tails))
        relation_tag_accuracy.update(torch.flatten(pred_relation_tags, end_dim = -2), torch.flatten(relation_tags))
      tb_writer.add_scalar('entity_start_accuracy', entity_start_accuracy.compute(), global_steps)
      tb_writer.add_scalar('entity_stop_accuracy', entity_stop_accuracy.compute(), global_steps)
      tb_writer.add_scalar('entity_tag_accuracy', entity_tag_accuracy.compute(), global_steps)
      tb_writer.add_scalar('relation_head_accuracy', relation_head_accuracy.compute(), global_steps)
      tb_writer.add_scalar('relation_tail_accuracy', relation_tail_accuracy.compute(), global_steps)
      tb_writer.add_scalar('relation_tag_accuracy', relation_tag_accuracy.compute(), global_steps)
      ckpt = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler}
      save(ckpt, join(FLAGS.ckpt, 'model-ep%d.pth' % epoch))

if __name__ == "__main__":
  add_options()
  app.run(main)

