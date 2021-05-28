import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from embeddings import Embedding

import torchsnooper
class SentEncoderConfig(object):
  def __init__(self):
    super().__init__()
    self.num_words = 10424
    self.dim_word = 300
    self.fix_word_embed = True
    self.rnn_type = 'gru' # gru, lstm
    self.bidirectional = True
    self.rnn_hidden_size = 1024
    self.num_layers = 1
    self.dropout = 0.5

  def _assert(self):
    assert self.rnn_type in ['gru', 'lstm'], 'invalid rnn_type'

class SentEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.embedding = Embedding(self.config.num_words, self.config.dim_word,
      fix_word_embed=self.config.fix_word_embed)
    dim_word = self.config.dim_word
    
    self.rnn = self.rnn_factory(self.config.rnn_type,
      input_size=dim_word, hidden_size=self.config.rnn_hidden_size, 
      num_layers=self.config.num_layers, dropout=self.config.dropout,
      bidirectional=self.config.bidirectional, bias=True, batch_first=True)
   
    self.dropout = nn.Dropout(self.config.dropout)
    self.init_weights()

  def init_weights(self):
    directions = ['']
    if self.config.bidirectional:
      directions.append('_reverse')
    for layer in range(self.config.num_layers):
      for direction in directions:
        for name in ['i', 'h']:
          weight = getattr(self.rnn, 'weight_%sh_l%d%s'%(name, layer, direction))
          nn.init.orthogonal_(weight.data)
          bias = getattr(self.rnn, 'bias_%sh_l%d%s'%(name, layer, direction))
          nn.init.constant_(bias, 0)
          if name == 'i' and self.config.rnn_type == 'lstm':
            bias.data.index_fill_(0, torch.arange(
              self.config.rnn_hidden_size, self.config.rnn_hidden_size*2).long(), 1)
    # print('self.rnn',self.rnn.state_dict()['weight_ih_l0'].shape)
    # print('self.rnn',self.rnn.state_dict()['weight_hh_l0'].shape)
    # print('self.rnn',self.rnn.state_dict()['bias_ih_l0'].shape)
    # print('self.rnn',self.rnn.state_dict()['bias_hh_l0'].shape)
          
  def forward_text_encoder(self, word_embeds, seq_lens, init_states):
     # outs.size = (batch, seq_len, num_directions * hidden_size)
    outs, states = self.calc_rnn_outs_with_sort(
      self.rnn, word_embeds, seq_lens, init_states)
    return outs
  # @torchsnooper.snoop()
  def sequence_mask(self,lengths, max_len=None, inverse=False):
    ''' Creates a boolean mask from sequence lengths.
    '''
    # lengths: LongTensor, (batch, )
    batch_size = lengths.size(0)
    max_len = max_len or lengths.max()
    mask = torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1)
    if inverse:
      mask = mask.ge(lengths.unsqueeze(1))
    else:
      mask = mask.lt(lengths.unsqueeze(1))
    return mask

  def calc_rnn_outs_with_sort(self, rnn, inputs, seq_lens, init_states=None):
    '''
    inputs: FloatTensor, (batch, seq_len, dim_ft)
    seq_lens: LongTensor, (batch,)
    init_states: FloatTensor, (num_layers * num_direction, batch, hidden_size)
    '''
    seq_len = inputs.size(1)
    # sort
    sorted_seq_lens, seq_sort_idx = torch.sort(seq_lens, descending=True)
    _, seq_unsort_idx = torch.sort(seq_sort_idx, descending=False)
    # pack
    inputs = torch.index_select(inputs, 0, seq_sort_idx)
    if init_states is not None:
      if isinstance(init_states, tuple):
        new_states = []
        for i, init_state in enumerate(init_states):
          new_states.append(torch.index_select(init_state, 1, seq_sort_idx))
        init_states = tuple(new_states)
      else:
        init_states = torch.index_select(init_states, 1, seq_sort_idx)
    
    packed_inputs = pack_padded_sequence(inputs, sorted_seq_lens, batch_first=True)
    # rnn
    packed_outs, states = rnn(packed_inputs, init_states)
    # unpack
    outs, _ = pad_packed_sequence(packed_outs, batch_first=True, 
      total_length=seq_len, padding_value=0)
    # unsort
    # outs.size = (batch, seq_len, num_directions * hidden_size)     
    outs = torch.index_select(outs, 0, seq_unsort_idx)   
    if isinstance(states, tuple):
      # states: (num_layers * num_directions, batch, hidden_size)
      new_states = []
      for i, state in enumerate(states):
        new_states.append(torch.index_select(state, 1, seq_unsort_idx))
      states = tuple(new_states)
    else:
      states = torch.index_select(states, 1, seq_unsort_idx)

    return outs, states


  def rnn_factory(self, rnn_type, **kwargs):
    # Use pytorch version when available.
    rnn = getattr(nn, rnn_type.upper())(**kwargs)
    return rnn

  # @torchsnooper.snoop()
  def forward(self, cap_ids, cap_lens, init_states=None, return_dense=False):
    '''
    Args:
      cap_ids: LongTensor, (batch, seq_len)
      cap_lens: FloatTensor, (batch, )
    Returns:
      if return_dense:
        embeds: FloatTensor, (batch, seq_len, embed_size)
      else:
        embeds: FloatTensor, (batch, embed_size)
    '''

    word_embeds = self.embedding(cap_ids)
    hiddens = self.forward_text_encoder(
      self.dropout(word_embeds), cap_lens, init_states)
    batch_size, max_seq_len, hidden_size = hiddens.size()

    if self.config.bidirectional:
      splited_hiddens = torch.split(hiddens, self.config.rnn_hidden_size, dim=2) 
      hiddens = (splited_hiddens[0] + splited_hiddens[1]) / 2

    if return_dense:
      return hiddens
    else:
      sent_masks = self.sequence_mask(cap_lens, max_seq_len, inverse=False).float()
      sent_embeds = torch.sum(hiddens * sent_masks.unsqueeze(2), 1) / cap_lens.unsqueeze(1).float()
      return sent_embeds


class SentAttnEncoder(SentEncoder):
  def __init__(self, config):
    super().__init__(config)
    self.ft_attn = nn.Linear(self.config.rnn_hidden_size, 1)
    self.softmax = nn.Softmax(dim=1)
  # @torchsnooper.snoop()
  def forward(self, cap_ids, cap_lens, init_states=None, return_dense=False):
    hiddens = super().forward(cap_ids, cap_lens, init_states=init_states, return_dense=True)

    attn_scores = self.ft_attn(hiddens).squeeze(2)
    cap_masks = self.sequence_mask(cap_lens, max_len=attn_scores.size(1), inverse=False)
    attn_scores = attn_scores.masked_fill(cap_masks == 0, -1e18)
    attn_scores = self.softmax(attn_scores)

    if return_dense:
      return hiddens, attn_scores
    else:
      return torch.sum(hiddens * attn_scores.unsqueeze(2), 1)
  
