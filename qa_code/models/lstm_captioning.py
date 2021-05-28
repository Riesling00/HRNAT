from __future__ import division
from __future__ import absolute_import
import torch
import sys
sys.path.append('/mnt/hdd4/leiyu/ssl/msvd_c3d_irv2/models/')
import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
from models import _CaptioningModel
from mlsent import RoleGraphEncoderConfig, RoleGraphEncoder
import numpy as np
import torchsnooper
import gumbel

# dir = '/mnt/hdd4/leiyu/all_dataset/HGR_T2V/MSVD/results/RET.released/mlmatch/vis.resnet152.pth.txt.bigru.16role.gcn.1L.attn.1024.loss.bi.af.embed.4.glove.init/model/epoch.18.th'
# dir = '/mnt/hdd4/leiyu/all_dataset/HGR_T2V/MSVD/results/RET.released/mlmatch/vis.i3d+irv2.pth.txt.bigru.16role.gcn.1L.attn.1024.loss.bi.af.embed.4.glove.init/model/epoch.10.th'
dir = '/mnt/hdd4/leiyu/all_dataset/HGR_T2V/MSVD/results/RET.released/mlmatch/vis.c3d+irv2.pth.txt.bigru.16role.gcn.1L.attn.1024.loss.bi.af.embed.4.glove.init/model/epoch.17.th' #9

class GumbelAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size):
        super(GumbelAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1, bias=False)

    def forward(self, feats, key):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        v = self.wv(feats)
       
        inputs = self.wh(key).unsqueeze(1).expand_as(v) + v
        outputs = self.wa(torch.tanh(inputs)).squeeze(-1)
        # if self.training:
        
        alpha = gumbel.st_gumbel_softmax(outputs)
        # else:
        # alpha = gumbel.greedy_select(outputs).float()

        att_feats = torch.bmm(alpha.unsqueeze(1), feats).squeeze(1)
        return att_feats, alpha

class LSTMCaptionModel(_CaptioningModel):
    def __init__(self, args, seq_len, vocab_size, bos_idx, h2_first_lstm=False, vid_second_lstm=False):
        super(LSTMCaptionModel, self).__init__(seq_len)
        self.bos_idx = bos_idx
        self.vocab_size = vocab_size
        self.feature_size = args.feature_size
        self.input_encoding_size = args.word_embed
        self.rnn_size = args.rnn_size
        self.feature_appearence = 2560
        self.feature_size_motion = 4096
        self.ss_prob = 0.0
        self.att_size = args.att_size
        self.h2_first_lstm = h2_first_lstm
        self.vid_second_lstm = vid_second_lstm

        self.embed = nn.Embedding(vocab_size, self.input_encoding_size) 
        # self.embed_ = nn.Sequential(self.embed,
        #                         nn.ReLU(),
        #                         nn.Dropout(args.dropout))

        self.img_embed_h_1 = nn.Linear(self.feature_appearence, self.rnn_size)  # (rnn_size, rnn_size)
        self.img_embed_c_1 = nn.Linear(self.feature_appearence, self.rnn_size)  # (rnn_size, rnn_size)
        self.img_embed_h_2 = nn.Linear(self.feature_appearence, self.rnn_size)  # (rnn_size, rnn_size)
        self.img_embed_c_2 = nn.Linear(self.feature_appearence, self.rnn_size)  # (rnn_size, rnn_size)

        self.img_embed_h_3 = nn.Linear(self.feature_appearence, self.feature_size)  # (rnn_size, rnn_size)
        self.img_embed_c_3 = nn.Linear(self.feature_appearence, self.feature_size)  # (rnn_size, rnn_size)

        self.W1_is = nn.Linear(self.feature_appearence, self.feature_size)

        #======== hierarchical ========#   
        self.ft_embed_1 = nn.Linear(self.feature_size_motion, self.feature_size, bias=True)
        self.ft_embed_2 = nn.Linear(self.feature_size, self.feature_size, bias=True)
        self.dropout_1 = nn.Dropout(p = 0.5)
        self.dropout_2 = nn.Dropout(p = 0.5)
        self.dropout_3 = nn.Dropout(p = 0.5)
        self.dropout_4 = nn.Dropout(p = 0.2)
        self.att_va = nn.Linear(self.feature_size, self.att_size, bias=False)
        self.att_ha = nn.Linear(self.rnn_size, self.att_size, bias=False)
        self.att_a = nn.Linear(self.att_size, 1, bias=False)

        self.att_vb = nn.Linear(self.feature_size, self.att_size, bias=False)
        self.att_hb = nn.Linear(self.rnn_size, self.att_size, bias=False)
        self.att_b = nn.Linear(self.att_size, 1, bias=False)
        # self.batchnorm_1 = nn.BatchNorm1d(30) 
        # self.batchnorm_2 = nn.BatchNorm1d() 
        # self.atten_embed_dropout = nn.Dropout(args.dropout)
        

        if self.h2_first_lstm:
            self.lstm_cell_1 = nn.LSTMCell(self.rnn_size + self.rnn_size + self.input_encoding_size, self.rnn_size)
            # self.lstm_cell_1 = nn.LSTMCell(motion_size + rgb_size + input_encoding_size +rnn_size, rnn_size)
        else:
            self.lstm_cell_1 = nn.LSTMCell(self.feature_size + self.input_encoding_size, self.rnn_size)


        if self.vid_second_lstm:
            self.lstm_cell_2 = nn.LSTMCell(self.rnn_size + self.feature_size + self.feature_size, self.rnn_size)
        elif args.pos_flag is True:
            # self.lstm_cell_2 = nn.LSTMCell(rnn_size + motion_size + rgb_size, rnn_size)
            self.lstm_cell_2 = nn.LSTMCell(self.rnn_size + self.rnn_size + self.rnn_size + self.rnn_size + self.rnn_size, self.rnn_size)
        else:
            self.lstm_cell_2 = nn.LSTMCell(self.rnn_size + self.rnn_size + self.rnn_size, self.rnn_size)

        self.W2_hs = nn.Linear(self.rnn_size, 768)
        self.vid_fn = nn.Linear(self.feature_size,self.rnn_size)

        self.dropout = nn.Dropout(args.dropout)
        self.out_fc = nn.Linear(768, vocab_size)
        # ==================== Stochastic Depth =========================#
        self.prob = [1.,0.5]
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.ss_depth = args.ss_depth
        #===================== matching =========================# 
        if args.matching_flag is True:
            self.config_txt = RoleGraphEncoderConfig()
            self.txt_embed = RoleGraphEncoder(self.config_txt)
            if args.matching_txt_load is True and args.fine_tune is False:
                fname = dir
                checkpoint = torch.load(fname)
                print('loading the best retrieval model --------')
                try:
                    checkpoint.train()
                except AttributeError as error:
                    print('error')
                
                self.txt_embed.load_state_dict(checkpoint['text_encoder'])
                if args.fixed is True:
                    for p in self.txt_embed.parameters():
                        p.requires_grad = False
                        print('retrivel parameters is requires_grad :', p.requires_grad) 
        
            # print('self.txt_embed',self.txt_embed.state_dict()['rnn.weight_ih_l0'].shape)
            # print('self.txt_embed',self.txt_embed.state_dict()['rnn.weight_hh_l0'].shape)
            # print('self.txt_embed',self.txt_embed.state_dict()['rnn.bias_ih_l0'].shape)
            # print('self.txt_embed',self.txt_embed.state_dict()['rnn.bias_hh_l0'].shape)
            self.margin = 0.2
            self.max_violation = True
            self.topk = 1
            self.direction = 'bi'
            self.num_verbs = 4
            self.num_nouns = 6
            self.simattn_sigma = 4
            self.attn_fusion = 'embed'      

        # ==================== reconstruction task ===================== #
        self.reconstructor_flag = args.reconstructor_flag
        if self.reconstructor_flag is True:
            self.W = nn.Linear(self.feature_size, self.att_size, bias=False)
            self.U = nn.Linear(self.att_size, self.rnn_size, bias=False)
            self.b = nn.Parameter(torch.Tensor(self.att_size).uniform_(-1,1), requires_grad=True)
            self.w = nn.Linear(self.rnn_size, 1, bias=False)
            self.lstm_reconstruction = nn.LSTMCell(self.att_size, self.feature_size)
        self.pos_flag = args.pos_flag
        if self.pos_flag is True:
            self.module_attn = GumbelAttention(self.rnn_size, self.rnn_size, self.rnn_size)
            self.func_hb = nn.Linear(self.rnn_size, self.att_size)
            self.func_b = nn.Linear(self.att_size, 1, bias = False)
            self.func_vb = nn.Linear(self.rnn_size, self.att_size)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.ft_embed_1.weight)
        nn.init.constant_(self.ft_embed_1.bias, 0)
        nn.init.xavier_normal_(self.ft_embed_2.weight)
        nn.init.constant_(self.ft_embed_2.bias, 0)

        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.out_fc.weight)
        nn.init.constant_(self.out_fc.bias, 0)

        nn.init.xavier_normal_(self.W1_is.weight)
        nn.init.constant_(self.W1_is.bias, 0)

        nn.init.xavier_normal_(self.W2_hs.weight)
        nn.init.constant_(self.W2_hs.bias, 0)

        nn.init.xavier_normal_(self.att_va.weight)
        nn.init.xavier_normal_(self.att_ha.weight)
        nn.init.xavier_normal_(self.att_a.weight)

        nn.init.xavier_normal_(self.att_vb.weight)
        nn.init.xavier_normal_(self.att_hb.weight)
        nn.init.xavier_normal_(self.att_b.weight)

        nn.init.xavier_normal_(self.lstm_cell_1.weight_ih)
        nn.init.orthogonal_(self.lstm_cell_1.weight_hh)
        nn.init.constant_(self.lstm_cell_1.bias_ih, 0)
        nn.init.constant_(self.lstm_cell_1.bias_hh, 0)

        nn.init.xavier_normal_(self.lstm_cell_2.weight_ih)
        nn.init.orthogonal_(self.lstm_cell_2.weight_hh)
        nn.init.constant_(self.lstm_cell_2.bias_ih, 0)
        nn.init.constant_(self.lstm_cell_2.bias_hh, 0)

        if self.reconstructor_flag is True:
            nn.init.xavier_normal_(self.W.weight)
            nn.init.xavier_normal_(self.w.weight)
            nn.init.xavier_normal_(self.U.weight)
            nn.init.xavier_normal_(self.lstm_reconstruction.weight_ih)
            nn.init.orthogonal_(self.lstm_reconstruction.weight_hh)
            nn.init.constant_(self.lstm_reconstruction.bias_ih, 0)
            nn.init.constant_(self.lstm_reconstruction.bias_hh, 0)

        if self.pos_flag is True:
            nn.init.xavier_normal_(self.func_hb.weight)
            nn.init.xavier_normal_(self.func_vb.weight)
            nn.init.xavier_normal_(self.func_b.weight)
            nn.init.constant_(self.func_hb.bias, 0)
            nn.init.constant_(self.func_vb.bias, 0)   


    def init_hidden(self, feat):
        feat_mask = (torch.sum(feat, -1, keepdim=True) != 0).float()
        feat_mean = torch.sum(feat, 1) / torch.sum(feat_mask, 1)  # (1,m,visual_size)
        state1 = (self.img_embed_h_1(feat_mean), self.img_embed_c_1(feat_mean))
        state2 = (self.img_embed_h_2(feat_mean), self.img_embed_c_2(feat_mean))
        state3 = (self.img_embed_h_3(feat_mean), self.img_embed_c_3(feat_mean))
        return (state1, state2), state3

    def forward_video_att(self, state_1, actions, entities):
        ######## Temporal attention #########
        actions = self.att_va(actions)
        vid_weights = torch.tanh(actions + self.att_ha(state_1[0]).unsqueeze(1)) #torch.Size([700, 30, 512])
        vid_weights = self.att_a(vid_weights) # torch.Size([700, 30, 1]
 
        att_weights = F.softmax(vid_weights, 1)  # (b_s, n_regions, 1) = torch.Size([700, 30, 1] 

        att_weights = self.feat_mask * att_weights
        att_weights = self.dropout_1(att_weights / torch.sum(att_weights, 1, keepdim=True))
        
        c_a_t = torch.sum(actions * att_weights, 1) # torch.Size([700, 512]) 

        ######## Spatial attention ########
        entities = self.att_vb(entities) 
        vid_local_weights = torch.tanh(entities + self.att_hb(state_1[0]).unsqueeze(1)) #torch.Size([700, 30, 512])
        vid_local_weights = self.att_b(vid_local_weights) # torch.Size([700, 30, 1]
 
        atten_local_weights = F.softmax(vid_local_weights, 1)  # (b_s, n_regions, 1) = torch.Size([700, 30, 1] 

        atten_local_weights = self.feat_mask * atten_local_weights
        atten_local_weights = self.dropout_2(atten_local_weights / torch.sum(atten_local_weights, 1, keepdim=True))
        c_o_t = torch.sum(entities * atten_local_weights, 1) # torch.Size([700, 512]) 
        return c_a_t, c_o_t

    def forward_local_reconstructor(self, decoder_hiddens, hidden, caption_masks):
        # attention_masks = caption_masks.transpose(0, 1)
        if self.ss_depth is True:
            if torch.equal(self.m.sample(),torch.ones(1)):
                self.W.weight.requires_grad = True
                self.U.weight.requires_grad = True
                self.b.weight.requires_grad = True
                self.w.weight.requires_grad = True
                self.lstm_reconstruction.weight_ih.requires_grad = True
                self.lstm_reconstruction.weight_hh.requires_grad = True
                self.lstm_reconstruction.bias_ih.requires_grad = True
                self.lstm_reconstruction.bias_hh.requires_grad = True
            else:
                # Resnet does not use bias terms
                self.W.weight.requires_grad = False
                self.U.weight.requires_grad = False
                self.b.weight.requires_grad = False
                self.lstm_reconstruction.weight_ih.requires_grad = False
                self.lstm_reconstruction.weight_hh.requires_grad = False
                self.lstm_reconstruction.bias_ih.requires_grad = False
                self.lstm_reconstruction.bias_hh.requires_grad = False

        attention_masks = caption_masks
        Wh = self.W(hidden[0])
        Uv = self.U(decoder_hiddens)
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        energies = self.w(torch.tanh(Wh + Uv + self.b))
        if attention_masks is not None:  
            energies = energies.squeeze(2)
            energies[~attention_masks] = -float('inf')
            energies = energies.unsqueeze(2)
        attn_weights = F.softmax(energies, dim=1)
        attn_weights = self.dropout(attn_weights)
        weighted_feats = decoder_hiddens * attn_weights # batch x step x att_size
        decoder_hidden = weighted_feats.sum(dim=1) # batch x att_size
        decoder_hidden = self.dropout(decoder_hidden)
        state = self.lstm_reconstruction(decoder_hidden, hidden) 
        return state
    # @torchsnooper.snoop()
    def step(self, t, state, prev_outputs, feats, seqs, previous_cells_, mode='teacher_forcing'):
        assert (mode in ['teacher_forcing', 'feedback'])
        feat, motion = feats
        b_s = feat.size(0)

        self.feat_mask = (torch.sum(feat, -1, keepdim=True) != 0).float() #torch.Size([700, 30, 1])
      
        # self.feat = feat
        feat = torch.tanh(self.W1_is(feat))
        self.feat = feat
        self.vid_descriptor = torch.sum(feat, 1) / torch.sum(self.feat_mask, 1) # [700,512]
  
        self.actions = self.dropout_1(self.ft_embed_1(motion))   #[batch, 30, 1024]
        self.entities = self.dropout_2(self.ft_embed_2(feat))

        state_1, state_2 = state  # state_1[0] torch.Size([700, 512]) x 2
        if mode == 'teacher_forcing':
            if t >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = feat.data.new(b_s).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                
                if sample_mask.sum() == 0:
                    it = seqs[:, t].clone()
          
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                
                    it = seqs[:, t].data.clone() #batch
                    prob_prev = torch.exp(prev_outputs.data)  # fetch prev distribution: shape Nx(M+1) 512 x 11160
                
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind)) #index_select(dim,index)
            else:
                it = seqs[:, t].clone()
                # <BOS>

        elif mode == 'feedback':
            if t == 0:
                it = feat.data.new_full((b_s,), self.bos_idx).long()
            else:
                it = prev_outputs.squeeze(1)
        
        xt = self.embed(it) #it [600] ->torch.Size([700, 300])
        
        if self.h2_first_lstm:
            input_1 = torch.cat([state_2[0], self.vid_descriptor, xt], 1)
        else:
            input_1 = torch.cat([self.vid_descriptor, xt], 1)

        state_1 = self.lstm_cell_1(input_1, state_1)
        
        c_a_t, c_o_t = self.forward_video_att(state_1, self.actions, self.entities) 

        if self.pos_flag:
            ################## func feats ##################
            previous_cells = previous_cells_
            if isinstance(previous_cells, list):
                previous_cells = torch.stack(previous_cells, dim=1)
            previous_cells = self.func_vb(previous_cells)
   
            func_weights = torch.tanh(previous_cells + self.func_hb(state_1[0]).unsqueeze(1)) #torch.Size([700, 30, 512])
            func_weights = self.func_b(func_weights) # torch.Size([700, 30, 1]
    
            atten_func_weights = F.softmax(func_weights, 1)  # (b_s, n_regions, 1) = torch.Size([700, 30, 1] 
            # atten_func_weights =  * atten_func_weights
            atten_func_weights = self.dropout_3(atten_func_weights / torch.sum(atten_func_weights, 1, keepdim=True))
            func_feats = torch.sum(previous_cells * atten_func_weights, 1) # torch.Size([700, 512]) 

        if mode == 'teacher_forcing' and self.pos_flag:
            _, self.module_weight = self.pos_module_selection(c_a_t, c_o_t, func_feats, state_1[0])

        self.vid_descriptor = self.vid_fn(self.vid_descriptor)
        self.vid_descriptor = self.dropout_4(self.vid_descriptor)
        if self.vid_second_lstm:
            input_2 = torch.cat([state_1[0], self.vid_descriptor, self.vid_descriptor], 1)
        elif self.pos_flag:
            input_2 = torch.cat([state_1[0], c_a_t, c_o_t, func_feats, self.vid_descriptor], 1)
        else:
            input_2 = torch.cat([state_1[0], c_a_t, c_o_t], 1)
        #input_2 torch.Size([700, 1024])
        state_2 = self.lstm_cell_2(input_2, state_2)

        if self.pos_flag:
            self.previous_cells.append(state_2[1])
            
        out_ = self.dropout(self.W2_hs(state_2[0])) #torch.Size([700, 768])

        out = F.log_softmax(self.out_fc(out_), dim=-1) #torch.Size([700, 11160])

        return out, (state_1, state_2)
  
    def matching(self, args, batch_data):    
        enc_outs = {
            'vid_sent_embeds': self.vid_descriptor,
            'vid_verb_embeds': self.actions, 
            'vid_noun_embeds': self.entities,
            'vid_lens': batch_data['attn_lens'],
        }
    
        cap_enc_outs = self.forward_text_embed(batch_data)
        enc_outs.update(cap_enc_outs)
        sent_scores, verb_scores, noun_scores = self.generate_scores(**enc_outs)
            
        scores = (sent_scores + verb_scores + noun_scores) / 3
        sent_loss = self.contrastiveLoss(sent_scores)
        verb_loss = self.contrastiveLoss(verb_scores)
        noun_loss = self.contrastiveLoss(noun_scores)
        fusion_loss = self.contrastiveLoss(scores) 

        loss = fusion_loss * args.vse_loss_weight
        return loss
    
    def cosine_sim(self,im, s):
        '''cosine similarity between all the image and sentence pairs
        '''

        inner_prod = im.mm(s.t())
        im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)
        s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
        sim = inner_prod / (im_norm * s_norm)
        return sim

    def generate_phrase_scores(self, vid_embeds, vid_masks, phrase_embeds, phrase_masks):
        '''Args:
        - vid_embeds: (batch, num_frames, embed_size)
        - vid_masks: (batch, num_frames)
        - phrase_embeds: (batch, num_phrases, embed_size)
        - phrase_masks: (batch, num_phrases)
        '''
        batch_vids, num_frames, _ = vid_embeds.size()
        vid_pad_masks = (vid_masks == 0).unsqueeze(1).unsqueeze(3)
        batch_phrases, num_phrases, dim_embed = phrase_embeds.size()

        # compute component-wise similarity
        vid_2d_embeds = vid_embeds.view(-1, dim_embed)
        phrase_2d_embeds = phrase_embeds.view(-1, dim_embed)
        # size = (batch_vids, batch_phrases, num_frames, num_phrases)
        ground_sims = self.cosine_sim(vid_2d_embeds, phrase_2d_embeds).view(
        batch_vids, num_frames, batch_phrases, num_phrases).transpose(1, 2)

        vid_attn_per_word = ground_sims.masked_fill(vid_pad_masks, 0)
        vid_attn_per_word[vid_attn_per_word < 0] = 0
        vid_attn_per_word = self.l2norm(vid_attn_per_word, dim=2)
        vid_attn_per_word = vid_attn_per_word.masked_fill(vid_pad_masks, -1e18)
        vid_attn_per_word = torch.softmax(self.simattn_sigma * vid_attn_per_word, dim=2)
        
        if self.attn_fusion == 'embed':
            vid_attned_embeds = torch.einsum('abcd,ace->abde', vid_attn_per_word, vid_embeds)
            word_attn_sims = torch.einsum('abde,bde->abd',
                self.l2norm(vid_attned_embeds),
                self.l2norm(phrase_embeds))
        elif self.attn_fusion == 'sim':
        # (batch_vids, batch_phrases, num_phrases)
            word_attn_sims = torch.sum(ground_sims * vid_attn_per_word, dim=2) 

        # sum: (batch_vid, batch_phrases)
        phrase_scores = torch.sum(word_attn_sims * phrase_masks.float().unsqueeze(0), 2) \
                    / torch.sum(phrase_masks, 1).float().unsqueeze(0).clamp(min=1)
        return phrase_scores

    def sequence_mask(self, lengths, max_len=None, inverse=False):
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

    def l2norm(self, inputs, dim=-1):
        # inputs: (batch, dim_ft)
        norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
        inputs = inputs / norm.clamp(min=1e-10)
        return inputs

    def generate_scores(self, **kwargs):
        ##### shared #####
        vid_lens = kwargs['vid_lens'] # (batch, )
        num_frames = kwargs['vid_verb_embeds'].size(1)
        vid_masks = self.sequence_mask(vid_lens, num_frames, inverse=False)

        ##### sentence-level scores #####
        sent_scores = self.cosine_sim(kwargs['vid_sent_embeds'], kwargs['sent_embeds'])

        ##### verb-level scores #####
        vid_verb_embeds = kwargs['vid_verb_embeds'] # (batch, num_frames, dim_embed)
        verb_embeds = kwargs['verb_embeds'] # (batch, num_verbs, dim_embed)
        verb_lens = kwargs['verb_lens'] # (batch, num_verbs)
        verb_masks = self.sequence_mask(torch.sum(verb_lens > 0, 1).long(), 
            self.num_verbs, inverse=False)
        # sum: (batch_vids, batch_sents)
        verb_scores = self.generate_phrase_scores(vid_verb_embeds, vid_masks, verb_embeds, verb_masks)

        ##### noun-level scores #####
        vid_noun_embeds = kwargs['vid_noun_embeds'] # (batch, num_frames, dim_embed)
        noun_embeds = kwargs['noun_embeds'] # (batch, num_nouns, dim_embed)
        noun_lens = kwargs['noun_lens'] # (batch, num_nouns)
        noun_masks = self.sequence_mask(torch.sum(noun_lens > 0, 1).long(), 
            self.num_nouns, inverse=False)
        # sum: (batch_vids, batch_sents)
        noun_scores = self.generate_phrase_scores(vid_noun_embeds, vid_masks, noun_embeds, noun_masks)
        return sent_scores, verb_scores, noun_scores

    def contrastiveLoss(self, scores, margin=None, average_batch=True):
        '''
        Args:
        scores: image-sentence score matrix, (batch, batch)
            the same row of im and s are positive pairs, different rows are negative pairs
        '''
        if margin is None:
            margin = self.margin

        batch_size = scores.size(0)
        diagonal = scores.diag().view(batch_size, 1) # positive pairs

        # mask to clear diagonals which are positive pairs
        pos_masks = torch.eye(batch_size).bool().to(scores.device)

        batch_topk = min(batch_size, self.topk)
        if self.direction == 'i2t' or self.direction == 'bi':
            d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence)
            # compare every diagonal score to scores in its collumn
            # caption retrieval
            cost_s = (margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill(pos_masks, 0)
            if self.max_violation:
                cost_s, _ = torch.topk(cost_s, batch_topk, dim=1)
                cost_s = cost_s / batch_topk
                if average_batch:
                    cost_s = cost_s / batch_size
            else:
                if average_batch:
                    cost_s = cost_s / (batch_size * (batch_size - 1))
            cost_s = torch.sum(cost_s)

        if self.direction == 't2i' or self.direction == 'bi':
            d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
            # compare every diagonal score to scores in its row
            cost_im = (margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill(pos_masks, 0)
            if self.max_violation:
                cost_im, _ = torch.topk(cost_im, batch_topk, dim=0)
                cost_im = cost_im / batch_topk
                if average_batch:
                    cost_im = cost_im / batch_size
            else:
                if average_batch:
                    cost_im = cost_im / (batch_size * (batch_size - 1))
            cost_im = torch.sum(cost_im)

        if self.direction == 'i2t':
            return cost_s
        elif self.direction == 't2i':
            return cost_im
        else:
            return cost_s + cost_im
    # @torchsnooper.snoop()
    def forward_text_embed(self, batch_data):
        sent_ids = batch_data['sent_ids']
        sent_lens = batch_data['sent_lens']
        verb_masks = batch_data['verb_masks']
        noun_masks = batch_data['noun_masks']
        node_roles = batch_data['node_roles']
        rel_edges = batch_data['rel_edges']
        verb_lens = torch.sum(verb_masks, 2)
        noun_lens = torch.sum(noun_masks, 2)
        # sent_embeds: (batch, dim_embed)
        # verb_embeds, noun_embeds: (batch, num_xxx, dim_embed)
        sent_embeds, verb_embeds, noun_embeds = self.txt_embed(
            sent_ids, sent_lens, verb_masks, noun_masks, node_roles, rel_edges)
        return {
        'sent_embeds': sent_embeds, 'sent_lens': sent_lens, 
        'verb_embeds': verb_embeds, 'verb_lens': verb_lens, 
        'noun_embeds': noun_embeds, 'noun_lens': noun_lens,
        }
        
    def pos_module_selection(self, c_a_t, c_o_t, func_feats, att_lstm_h):
        loc_feats = c_o_t
        rel_feats = c_a_t
        ################## func feats ##################
        # previous_cells = self.previous_cells
        # if isinstance(previous_cells, list):
        #     previous_cells = torch.stack(previous_cells, dim=1)
        # previous_cells = self.func_vb(previous_cells) 
        # func_weights = torch.tanh(previous_cells + self.func_hb(att_lstm_h).unsqueeze(1)) #torch.Size([700, 30, 512])
        # func_weights = self.func_b(func_weights) # torch.Size([700, 30, 1]
 
        # atten_func_weights = F.softmax(func_weights, 1)  # (b_s, n_regions, 1) = torch.Size([700, 30, 1] 
        # # atten_func_weights =  * atten_func_weights
        # atten_func_weights = self.dropout_2(atten_func_weights / torch.sum(atten_func_weights, 1, keepdim=True))
        # func_feats = torch.sum(previous_cells * atten_func_weights, 1) # torch.Size([700, 512]) 

        # feats = self.dropout_4(torch.stack([loc_feats, rel_feats, func_feats], dim=1))
        feats = torch.stack([loc_feats, rel_feats, func_feats], dim=1)
        feats, module_weight = self.module_attn(feats, att_lstm_h)
        
        return feats, module_weight