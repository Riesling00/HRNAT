import torch
import utils
import torchsnooper

class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _expand_state(self, selected_beam, cur_beam_size): 
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    # @torchsnooper.snoop()
    def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
 
        if isinstance(visual, torch.Tensor):     
            visual_shape = visual.shape #torch.Size([140, 512]) batch_size=700 dim=512   
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]  #(140,1,512)     
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:] #(700, 512)   
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2)) #(140, 5, 1)
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:] #(140, 5, 512)   
            visual_exp = visual.view(visual_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size) #torch.Size([140, 5, 512])  
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape) #torch.Size([700, 512])
        else:
            new_visual = []
            for im in visual:
                # print('##########################################################')
                visual_shape = im.shape
                # print('visual_shape',visual_shape)
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                # print('visual_exp_shape',visual_exp_shape)
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                # print('visual_red_shape',visual_red_shape)
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
                # print('selected_beam_red_size',selected_beam_red_size)
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
                # print('selected_beam_exp_size',selected_beam_exp_size)
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
                # print('selected_beam_exp',selected_beam_exp.shape,selected_beam_exp)
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
            # print('visual',visual.shape,visual)
        return visual

    def apply(self, state, visual, out_size=1, return_probs=False, **kwargs):
        img = visual[0]
        self.b_s = utils.get_batch_size(img)
        self.device = utils.get_device(img)
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []
        self.previous_cells = []
        outputs = []
        # visual torch.Size([140, 30, 1024])
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                self.previous_cells.append(state[1][1]) 
                visual, outputs, state = self.iter(t, state, visual, outputs, return_probs, **kwargs)
                #visual torch.Size([700, 30, 1024])
        # Sort result
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                                          self.max_len,
                                                                                          all_log_probs.shape[-1]))
        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

    def select(self, t, candidate_logprob, **kwargs):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        return selected_idx, selected_logprob
    # @torchsnooper.snoop()
    def _expandstate(self, input, cur_beam_size, selected_beam):
        new_input = []
        for i, s in enumerate(input):
            if isinstance(input, tuple) or isinstance(input, list):
                new_state_i = []
                for ii, ss in enumerate(s):
                
                    new_state_ii = self._expand_visual(ss, cur_beam_size, selected_beam)
            
                    new_state_i.append(new_state_ii)
                new_input.append(tuple(new_state_i))
            else:
                new_state_i = self._expand_visual(ss, cur_beam_size,selected_beam)
                new_input.append(new_state_i)
        return list(new_input)


    
    def iter(self, t: int, state, visual, outputs, return_probs, **kwargs):
      
        cur_beam_size = 1 if t == 0 else self.beam_size
        if t==0:
            x =state
        word_logprob, state = self.model.step(t, state, self.selected_words, visual, None, self.previous_cells, mode='feedback', **kwargs)
        
        #word_logprob torch.Size([700, 11160]) state[0][0]torch.Size([700, 512]) 
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)
        candidate_logprob = self.seq_logprob + word_logprob #torch.Size([140, cur_beam_size, 11160])
  
        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1) #torch.Size([140, 5, 1])
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous() #torch.Size([140, cur_beam_size, 11160])
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)
   
        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs) #torch.Size([140, 5])
        selected_beam = selected_idx / candidate_logprob.shape[-1] #torch.Size([140, 5])
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1] #torch.Size([140, 5])

        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))
     
        state = self._expandstate(state, cur_beam_size, selected_beam)
        if t == 0:
            self.previous_cells = []
            x = self._expandstate(x, cur_beam_size, selected_beam)
            self.previous_cells.append(x[1][1])
        img, motion = visual
        img = self._expand_visual(img, cur_beam_size, selected_beam)
        motion = self._expand_visual(motion, cur_beam_size, selected_beam)  #torch.Size([140, 30, 1024]) -> torch.Size([700, 30, 1024])
        visual = img, motion
        self.seq_logprob = selected_logprob.unsqueeze(-1)
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
        outputs.append(selected_words.unsqueeze(-1))
        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        self.log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs)
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)

        return visual, outputs, state
