import torch
from torch import distributions
import utils
from models.containers import Module
from models.beam_search import *
import torch.nn.functional as F
import torch.nn as nn 
from torch.autograd import Variable 

PAD = 0
class CaptioningModel(Module):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        super(CaptioningModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def step(self, t, prev_output, visual, visual2, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError

    def matching(self, args, batch_data, **kwargs):
        raise NotImplementedError

    def forward(self, args, batch_data):
        device = torch.device('cuda')
        images, motions, seq = batch_data['attn_fts'], batch_data['motion_fts'], batch_data['sent_ids_caption']
        seq_len = seq.size(1)
        feat_len = images.size(1)

        state, hidden = self.init_hidden(images)
        out = None
        loss_match = 0
        loss_re = 0.
        loss_pos = 0.
        outputs = []
        decoder_hiddens = []
        module_weights = []
        self.previous_cells = []
        self.previous_cells.append(state[1][1])

        for t in range(seq_len):
            out, state = self.step(t, state, out, (images,motions), seq, self.previous_cells, mode='teacher_forcing')
            outputs.append(out) 
            if args.reconstructor_flag:
                decoder_hiddens.append(state[1][0]) 
            if args.pos_flag:
                module_weights.append(self.module_weight)
        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1) 
        if args.reconstructor_flag:
            decoder_hiddens = torch.cat([o.unsqueeze(1) for o in decoder_hiddens], 1) 

        if args.matching_flag:
            loss_match = self.matching(args, batch_data) 
            # loss_match = loss_match_.detach_()

        if args.pos_flag is True and args.backpos is True:
            bsz = len(seq)
            pos_tags = batch_data['pos_tag']

            module_weights = torch.cat([o.unsqueeze(1) for o in module_weights], 1)
            module_weights = torch.cat([module_weights[j][: batch_data['sent_lens'][j]] for j in range(bsz)], 0)
            module_weights = module_weights.view(-1, 3)
    
            # remove pad and flatten pos_tags
            pos_tags = torch.cat([pos_tags[j][: batch_data['sent_lens'][j]] for j in range(bsz)], 0)
    
            pos_tags = pos_tags.view(-1)

            # compute linguistic loss
            loss_pos = nn.CrossEntropyLoss()(module_weights, pos_tags) * args.pos_lambda
        
        if args.reconstructor_flag is True:
            feats_recons = torch.zeros(images.size(0), feat_len, hidden[0].size(1)).to(device)
            captions = None
            if captions is None: 
                 _, captions = outputs.max(dim=2)

            caption_masks = (captions != PAD)
    
            for t in range(feat_len):
                hidden = self.forward_local_reconstructor(decoder_hiddens, hidden, caption_masks) 
                feats_recons[:, t, :] = hidden[0]
            loss_re = args.recon_lambda * F.mse_loss(self.feat, feats_recons)

        return outputs, loss_match, loss_re, loss_pos

    def test(self, visual: utils.TensorOrSequence, max_len: int, eos_idx: int,
             **kwargs) -> utils.Tuple[torch.Tensor, torch.Tensor]:
        b_s = utils.get_batch_size(visual)
        device = utils.get_device(visual)
        outputs = []
        log_probs = []

        mask = torch.ones((b_s,), device=device)
        with self.statefulness(b_s):
            out = None
            for t in range(max_len):
                log_probs_t = self.step(t, out, visual, None, mode='feedback', **kwargs)
                out = torch.max(log_probs_t, -1)[1]
                mask = mask * (out.squeeze(-1) != eos_idx).float()
                log_probs.append(log_probs_t * mask.unsqueeze(-1).unsqueeze(-1))
                outputs.append(out)

        return torch.cat(outputs, 1), torch.cat(log_probs, 1)

    def sample_rl(self, visual: utils.TensorOrSequence, visual2: utils.TensorOrSequence, max_len: int, **kwargs) -> \
    utils.Tuple[torch.Tensor, torch.Tensor]:
        b_s = utils.get_batch_size(visual)
        outputs = []
        log_probs = []

        with self.statefulness(b_s):
            out = None
            for t in range(max_len):
                out = self.step(t, out, visual, visual2, None, mode='feedback', **kwargs)
                distr = distributions.Categorical(logits=out[:, 0])
                out = distr.sample().unsqueeze(1)
                outputs.append(out)
                log_probs.append(distr.log_prob(out).unsqueeze(1))

        return torch.cat(outputs, 1), torch.cat(log_probs, 1)

    def beam_search(self, visual, max_len: int, eos_idx: int,
                    beam_size: int, out_size=1, return_probs=False, **kwargs):
        bs = BeamSearch(self, max_len, eos_idx, beam_size)
        # device = visual.device
        # b_s = visual.size(0)
        state, _ = self.init_hidden(visual[0])
        
        # state = self.init_state(b_s, device)
        return bs.apply(state, visual, out_size, return_probs, **kwargs)
