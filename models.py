#!/usr/bin/python3

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn, device
from torch.nn import functional as F
from transformers import AutoTokenizer, BartModel

class NERE(nn.Module):
  def __init__(self, entity_tag_num, relation_tag_num, max_entity_num = 10, max_relation_num = 10):
    super(NERE, self).__init__()
    self.entity_tag_num = entity_tag_num
    self.relation_tag_num = relation_tag_num
    self.max_entity_num = max_entity_num
    self.max_relation_num = max_relation_num
    self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    self.encoder_and_entity_decoder = BartModel.from_pretrained('facebook/bart-base')
    self.relation_decoder = BartModel.from_pretrained('facebook/bart-base').decoder
    self.entity_embed = nn.Embedding(num_embeddings = max_entity_num, embedding_dim = self.encoder_and_entity_decoder.config.d_model)
    self.entity_start = nn.Linear(self.encoder_and_entity_decoder.config.d_model, self.encoder_and_entity_decoder.config.max_position_embeddings)
    self.entity_end = nn.Linear(self.encoder_and_entity_decoder.config.d_model, self.encoder_and_entity_decoder.config.max_position_embeddings)
    self.entity_tag = nn.Linear(self.encoder_and_entity_decoder.config.d_model, entity_tag_num + 1)
    self.relation_embed = nn.Embedding(num_embeddings = max_relation_num, embedding_dim = self.encoder_and_entity_decoder.config.d_model)
    self.relation_head = nn.Linear(self.encoder_and_entity_decoder.config.d_model, max_entity_num)
    self.relation_tail = nn.Linear(self.encoder_and_entity_decoder.config.d_model, max_entity_num)
    self.relation_tag = nn.Linear(self.encoder_and_entity_decoder.config.d_model, relation_tag_num + 1)
  def forward(self, input_ids, attention_mask):
    # 1) entity prediction
    entity_embed_inputs = torch.tile(torch.unsqueeze(torch.range(0, self.max_entity_num - 1, dtype = torch.int32), dim = 0), (input_ids.shape[0], 1)) # entity_embed_inputs.shape = (batch, max_entity_num)
    entity_embed_inputs = entity_embed_inputs.to(self.encoder_and_entity_decoder.device)
    entity_embed_inputs = self.entity_embed(entity_embed_inputs) # entity_embed_inputs.shape = (batch, max_entity_num, d_model)
    outputs = self.encoder_and_entity_decoder(input_ids = input_ids, attention_mask = attention_mask, decoder_inputs_embeds = entity_embed_inputs)
    encoder_outputs = outputs.encoder_last_hidden_state
    decoder_outputs = outputs.last_hidden_state # last_hidden_state.shape = (batch, max_length, hidden_dim)
    # entity start
    entity_start = self.entity_start(decoder_outputs)
    # entity end
    entity_end = self.entity_end(decoder_outputs)
    # entity tag
    entity_tag = self.entity_tag(decoder_outputs)
    entity_start_idx = torch.argmax(entity_start, dim = -1) # entity_start_idx.shape = (batch, max_entity_num)
    entity_end_idx = torch.argmax(entity_end, dim = -1) # entity_end_idx.shape = (batch, max_entity_num)
    entity_tag_idx = torch.argmax(entity_tag, dim = -1) # entity_tag_idx.shape = (batch, max_entity_num)
    entity_mask = entity_tag_idx < self.entity_tag_num # entity_mask.shape = (batch, max_entity_num)
    valid_mask = entity_start_idx < entity_end_idx # valid_entity.shape = (batch, max_entity_num)
    mask = torch.logical_and(entity_mask, valid_mask) # mask.shape = (batch, max_entity_num)
    batch_entities_hidden = list()
    batch_entities_mask = list()
    for hidden, start, end, mask in zip(encoder_outputs, entity_start_idx, entity_end_idx, mask):
      start = torch.masked_select(start, mask) # start.shape = (entity_num,)
      end = torch.masked_select(end, mask) # end.shape = (entity_num,)
      entities_hidden = [torch.mean(hidden[s:e], dim = 0) for s, e in zip(start, end)]
      entities_hidden = torch.stack(entities_hidden, dim = 0) if len(entities_hidden) else torch.zeros(0, hidden.shape[-1]).to(hidden.device) # entities_hidden.shape = (entity_num, hidden_dim)
      batch_entities_hidden.append(torch.cat([entities_hidden, torch.ones((self.max_entity_num - entities_hidden.shape[0], entities_hidden.shape[1])).to(entities_hidden.device)], dim = 0))
      attention_mask = torch.cat([torch.ones(entities_hidden.shape[0]), torch.zeros(self.max_entity_num - entities_hidden.shape[0])], dim = 0).to(entities_hidden.device)
      # to prevent softmax of self.relation_decoder from yielding nan
      attention_mask = torch.cat([torch.zeros(self.max_entity_num - 1), torch.ones(1)], dim = 0).to(entities_hidden.device) if not torch.any(attention_mask) else attention_mask
      batch_entities_mask.append(attention_mask)
    entities_hidden = torch.stack(batch_entities_hidden) # entities_hidden.shape = (batch, max_entity_num, hidden_dim)
    entities_mask = torch.stack(batch_entities_mask) # entities_mask.shape = (batch, max_entity_num)
    # 2) relationship prediction
    relation_embed_inputs = torch.tile(torch.unsqueeze(torch.range(0, self.max_relation_num - 1, dtype = torch.int32), dim = 0), (input_ids.shape[0], 1)) # relation_embed_inputs.shape = (batch, max_relation_num)
    relation_embed_inputs = relation_embed_inputs.to(self.encoder_and_entity_decoder.device)
    relation_embed_inputs = self.relation_embed(relation_embed_inputs) # relation_embed_inputs.shape = (batch, max_relation_num, d_model)
    outputs = self.relation_decoder(encoder_hidden_states = entities_hidden, encoder_attention_mask = entities_mask, inputs_embeds = relation_embed_inputs)
    decoder_outputs = outputs.last_hidden_state
    # relation head
    relation_head = self.relation_head(decoder_outputs)
    # relation tail
    relation_tail = self.relation_tail(decoder_outputs)
    # relation tag
    relation_tag = self.relation_tag(decoder_outputs)
    return entity_start, entity_end, entity_tag, relation_head, relation_tail, relation_tag

class HungarianMatcher(nn.Module):
  def __init__(self, entity_types):
    super(HungarianMatcher, self).__init__()
    self.entity_types = entity_types
  @torch.no_grad()
  def forward(self, start_pred, end_pred, tag_pred, start_label, end_label, tag_label):
    # start_pred.shape = (batch, max_entity_num, max_seq_len)
    # end_pred.shape = (batch, max_entity_num, max_seq_len)
    # tag_pred.shape = (batch, max_entity_num, class_num)
    # start_label.shape = (batch, max_entity_num)
    # end_label.shape = (batch, max_entity_num)
    # tag_label.shape = (batch, max_entity_num)
    start_pred = torch.argmax(start_pred, dim = -1)
    end_pred = torch.argmax(end_pred, dim = -1)
    span_pred = torch.cat([start_pred, end_pred], dim = -1) # span_pred.shape = (batch, max_entity_num, 2)
    span_label = torch.stack([start_label, end_label], dim = -1) # span_label.shape = (batch, max_entity_num, 2)
    assignments = list()
    for span_p, tag_p, span_l, tag_l in zip(span_pred, tag_pred, span_label, tag_label):
      mask_p = tag_p < len(self.entity_types)
      mask_l = tag_l < len(self.entity_types)
      span_p = span_p[mask_p,:] # span_p.shape = (pred_target_num, 2)
      span_l = span_l[mask_l,:] # span_l.shape = (label_target_num, 2)
      # 1) span loss
      span_p = torch.unsqueeze(span_p, dim = -2) # span_p.shape = (pred_target_num, 1, 2)
      span_l = torch.unsqueeze(span_l, dim = -3) # span_l.shape = (1, label_target_num, 2)
      span_loss = torch.cdist(span_p, span_l, p = 1) # span_loss.shape = (pred_target_num, label_target_num,1)
      span_loss = torch.squeeze(span_loss, dim = -1) # span_loss.shape = (pred_target_num, label_target_num)
      # 2) iou loss
      span_p_left, _ = torch.min(span_p, dim = -1) # span_p_left.shape = (pred_target_num, 1)
      span_p_right, _ = torch.max(span_p, dim = -1) # span_p_right.shape = (pred_target_num, 1)
      span_l_left, _ = torch.min(span_l, dim = -1) # span_l_left.shape = (1, label_target_num)
      span_l_right, _ = torch.max(span_l, dim = -1) # span_l_right.shape = (1, label_target_num)
      intersect_left = torch.maximum(span_p_left, span_l_left) # intersect_left.shape = (pred_target_num, label_target_num)
      intersect_right = torch.minimum(span_p_right, span_p_right) # intersect_right.shape = (pred_target_num, label_target_num)
      intersect = torch.maximum(intersect_right - intersect_left, 0) # intersect.shape = (pred_target_num, label_target_num)
      union_left = torch.minimum(span_p_left, span_l_left) # union_left.shape = (pred_target_num, label_target_num)
      union_right = torch.maximum(span_p_right, span_l_right) # union_right.shape = (pred_target_num, label_target_num)
      union = torch.maximum(union_right - union_right, 1e-10) # union.shape = (pred_target_num, label_target_num)
      iou_loss = -intersect / union # iou.shape = (pred_target_num, label_target_num)
      # 3) class loss
      class_loss = -tag_p[:,tag_label] # class_loss.shape = (pred_target_num, label_target_num)
      cost = span_loss + iou_loss + class_loss
      i,j = linear_sum_assignment(cost.cpu().numpy())
      assignments.append((torch.as_tensor(i, dtype = torch,int64), torch.as_tensor(j, dtype = torch.int64)))
    return assignments

class Criterion(nn.Module):
  def __init__(self, entity_types):
    super(Criterion, self).__init__()
    self.entity_types = entity_types
    self.matcher = HungarianMatcher(entity_types)
    self.criterion = nn.CrossEntropyLoss()
  def forward(self, start_pred, end_pred, tag_pred, start_label, end_label, tag_label):
    indices = self.matcher(start_pred, end_pred, tag_pred, start_label, end_label, tag_label)
    loss = list()
    for s_p, e_p, t_p, s_l, e_l, t_l, (i, j) in zip(start_pred, end_pred, tag_pred, start_label, end_label, tag_label, indices):
      mask_p = t_p < len(self.entity_types)
      mask_l = t_l < len(self.entity_types)
      s_p = s_p[mask_p,:][i] # s_p.shape = (pred_target_num, max_seq_len)
      e_p = e_p[mask_p,:][i] # e_p.shape = (pred_target_num, max_seq_len)
      t_p = t_p[mask_p][i] # t_p.shape = (pred_target_num, class_num)
      s_l = s_l[mask_l][j] # s_l.shape = (label_target_num)
      e_l = e_l[mask_l][j] # e_l.shape = (label_target_num)
      t_l = t_l[mask_l][j] # t_l.shape = (label_target_num)
      loss1 = self.criterion(s_p.transpose(1,-1), s_l)
      loss2 = self.criterion(e_p.transpose(1,-1), e_l)
      loss3 = self.criterion(t_p.transpose(1,-1), t_l)
      loss.append(loss1 + loss2 + loss3)
    return torch.mean(loss), indices

if __name__ == "__main__":
  d = 'cuda'
  model = NERE(entity_tag_num = 7, relation_tag_num = 5, max_entity_num = 14, max_relation_num = 10).to(device(d))
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', add_prefix_space = True)
  inputs = tokenizer(["Hello, my dog is cute", "Hello the world!"], return_tensors = 'pt', padding = True)
  es,ee,et,rh,rt,rt = model(inputs['input_ids'].to(device(d)), inputs['attention_mask'].to(device(d)))
  print(es.shape,ee.shape,et.shape,rh.shape,rt.shape,rt.shape)
  hidden = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str('Hello, my dog is cute')
  print(hidden)
