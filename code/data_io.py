# coding: utf-8

import json
import torch
import torch.utils.data as data
import os
import h5py
import numpy as np
# torch.multiprocessing.set_start_method('spawn', force=True)
BOS, EOS, UNK, PAD = 0, 1, 2, 10423 #7136 5517 4642
word2int_file = '/mnt/hdd4/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/word2int.json'
def get_sub_frames(frames, K):
    # from all frames, take K of them, then add end of video frame
    lens = min(len(frames),K)
    if len(frames) < K:
        temp_zeros = np.zeros([K-frames.shape[0], frames.shape[1]])
        frames_ = np.concatenate((frames,temp_zeros), axis=0)
    else:
        index = np.linspace(0, len(frames), K, endpoint=False, dtype=int)  
        frames_ = frames[index]
    return frames_, lens

ROLES = ['V', 'ARG1', 'ARG0', 'ARG2', 'ARG3', 'ARG4',
 'ARGM-LOC', 'ARGM-MNR', 'ARGM-TMP', 'ARGM-DIR', 'ARGM-ADV', 
 'ARGM-PRP', 'ARGM-PRD', 'ARGM-COM', 'ARGM-MOD', 'NOUN'] 

class DictDataset(data.Dataset):
    #用于加载train和val集
    def __init__(self, dataset, feat_name, split, vid_max_len):
        self.vid_max_len = vid_max_len
        self.dataset = dataset
        if self.dataset == 'VATEX':
            self.data = json.load(open('/mnt/hdd4/leiyu/%s/wu/data/%s_data.json' % (dataset, split), 'rb'))
            self.role = json.load(open('/mnt/hdd4/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/ref_captions.json'))
            self.pos = json.load(open('/mnt/hdd4/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/pos_tag.json'))

        # self.feat_path = h5py.File('/hdd1/zengpengpeng/dataset/MSR_VTT/msrvtt_i3d_flow/feats.hdf5', 'r')
        if split in ['train','val']:
            self.feat_path = '/mnt/hdd4/leiyu/%s/val' %(dataset)
        else:
            self.feat_path = '/mnt/hdd4/leiyu/%s/public_test' % (dataset)


        # msrvtt_inpRes_rgb

    def __getitem__(self, index):
        data = self.data[index]#.type(torch.LongTensor)
        vid_id = data['vid_id']
        name = vid_id
        # cap = torch.tensor(data['caption'])
        cap = data['gt']

        vid_id = vid_id + '.npy'
        feat = np.load(open(os.path.join(self.feat_path, vid_id), 'rb'))
        feature, _ = get_sub_frames(feat.squeeze(), self.vid_max_len)
        feature = torch.from_numpy(feature).float()
        return feature, cap, name 

    def __len__(self):
        return len(self.data)


def collate_fn_dict(batch): # batch: ( data, cap, feat)
    feature, cap, name = zip(*batch)  # gts:
    feats = torch.stack(feature, dim=0)
    caps = []
    for w in cap:  #  for each data in the batch:
        gt = []
        for v in w:
            gt.append(v)
        caps.append(gt)
    return feats, caps, name # feat:(m,28,1536)


class TrainValDataset(data.Dataset):
    #用于加载train和val集
    def __init__(self, args, dataset, feat_name, split, vid_max_len):
        self.vid_max_len = vid_max_len
        self.dataset = dataset
        self.word2int = json.load(open(word2int_file))
        if self.dataset == 'VATEX':
            self.data = json.load(open('/mnt/hdd4/leiyu/%s/wu/data/%s_data.json' % (dataset, split), 'rb'))
            self.role = json.load(open('/mnt/hdd4/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/ref_captions.json'))
            self.pos = json.load(open('/mnt/hdd4/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/pos_tag.json'))

        # if split in ['train','val']:
        #     # self.feat_path = '/mnt/hdd1/leiyu/all_dataset/HGR_T2V/MSVD/ordered_feature/trn'
        #     # self.feat_path = h5py.File('/mnt/hdd1/leiyu/all_dataset/HGR_T2V/MSVD/ordered_feature/SA/resnet152.pth/trn_ft.hdf5','r')
        #     # self.feat_path_c3d = '/mnt/hdd4/leiyu/all_dataset/VATEX/c3d' #com
        self.feat_path_i3d = '/mnt/hdd4/leiyu/VATEX/val'
        # self.feat_path_c3d = '/mnt/hdd4/leiyu/VATEX/val'
        self.feat_path= '/mnt/hdd4/leiyu/all_dataset/VATEX/irv2' #com
            # self.feat_path= '/mnt/hdd4/leiyu/VATEX/val'


        self.matching_flag = args.matching_flag
        self.max_words_in_sent = args.max_words_in_sent  
        if True:
            self.num_verbs = args.num_verbs
            self.num_nouns = args.num_nouns
            self.role2int = {}
            for i, role in enumerate(ROLES):
                self.role2int[role] = i
                self.role2int['C-%s'%role] = i
                self.role2int['R-%s'%role] = i
            self.ref_graphs = json.load(open(args.ref_graph_file))

    def get_caption_outs(self, out, sent, graph):
        graph_nodes, graph_edges = graph
        # print(graph_nodes)
        # print(graph_edges)

        verb_node2idxs, noun_node2idxs = {}, {}
        edges = []
        out['node_roles'] = np.zeros((self.num_verbs + self.num_nouns, ), np.int32)

        # root node
        sent_ids, _, sent_len = self.process_sent(sent, self.max_words_in_sent)
        out['sent_ids'] = sent_ids
        out['sent_lens'] = sent_len

        # graph: add verb nodes
        node_idx = 1
        out['verb_masks'] = np.zeros((self.num_verbs, self.max_words_in_sent), np.bool)
        for knode, vnode in graph_nodes.items():
            k = node_idx - 1
            if k >= self.num_verbs:
                break
            if vnode['role'] == 'V' and np.min(vnode['spans']) < self.max_words_in_sent:
                verb_node2idxs[knode] = node_idx
                for widx in vnode['spans']:
                    if widx < self.max_words_in_sent:
                        out['verb_masks'][k][widx] = True
                out['node_roles'][node_idx - 1] = self.role2int['V']
                # add root to verb edge
                edges.append((0, node_idx))
                node_idx += 1
            
        # graph: add noun nodes
        node_idx = 1 + self.num_verbs
        out['noun_masks'] = np.zeros((self.num_nouns, self.max_words_in_sent), np.bool)
        for knode, vnode in graph_nodes.items():
            k = node_idx - self.num_verbs - 1
            if k >= self.num_nouns:
                break
            if vnode['role'] not in ['ROOT', 'V'] and np.min(vnode['spans']) < self.max_words_in_sent:
                noun_node2idxs[knode] = node_idx
                for widx in vnode['spans']:
                    if widx < self.max_words_in_sent:
                        out['noun_masks'][k][widx] = True
                out['node_roles'][node_idx - 1] = self.role2int.get(vnode['role'], self.role2int['NOUN'])
                node_idx += 1

        # graph: add verb_node to noun_node edges
        for e in graph_edges:
            if e[0] in verb_node2idxs and e[1] in noun_node2idxs:
                edges.append((verb_node2idxs[e[0]], noun_node2idxs[e[1]]))
                edges.append((noun_node2idxs[e[1]], verb_node2idxs[e[0]]))

        num_nodes = 1 + self.num_verbs + self.num_nouns
        rel_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for src_nodeidx, tgt_nodeidx in edges:
            rel_matrix[tgt_nodeidx, src_nodeidx] = 1
        # row norm
        for i in range(num_nodes):
            s = np.sum(rel_matrix[i])
            if s > 0:
                rel_matrix[i] /= s

        out['rel_edges'] = rel_matrix
        return out

    def process_sent(self, sent, max_words):
        tokens = [self.word2int.get(w, UNK) for w in sent.split()]
        # # add BOS, EOS?
        tokens_cap = [BOS] + tokens + [EOS]
        tokens_cap = tokens_cap[:max_words]
        tokens_len_cap = len(tokens_cap)
        
        tokens = tokens[:max_words]
        tokens_len = len(tokens)

        tokens = np.array(tokens + [EOS] * (max_words - tokens_len))

        return tokens, tokens_cap, tokens_len

    def process_pos(self, sent, max_words):


        sent = sent[:max_words]

        tokens_len = len(sent)

        return sent

    def __getitem__(self, index):
        data = self.data[index]#.type(torch.LongTensor)
        idx = data['idx']
        vid_id = data['vid_id']


        sent = self.role[vid_id][idx]
        vid_name = vid_id
        tag = self.pos[sent]
        vid_id = vid_id + '.npy'
        
        feat = np.load(open(os.path.join(self.feat_path, vid_id), 'rb'))
        if len(feat.shape) == 1:
            feat = np.expand_dims(feat, axis=0)
        feat, attn_lens = get_sub_frames(feat, self.vid_max_len)
        # feat_motion = np.load(open(os.path.join(self.feat_path_i3d, vid), 'rb'))
        feat_motion = np.load(open(os.path.join(self.feat_path_i3d, vid_id), 'rb'))
        motion, _ = get_sub_frames(feat_motion.squeeze(), self.vid_max_len)
        feat = torch.from_numpy(feat).float()
        motion = torch.from_numpy(motion).float()

        out = {}
        out['names'] = vid_name
        tag = self.process_pos(tag, self.max_words_in_sent)
        out['pos_tag'] = tag
        # out['sent_ids_caption'] = torch.tensor(cap) 
        out['attn_fts'] = feat
        out['motion_fts'] = motion
        # out['sent_lens'] = sent_len
        out['attn_lens'] = attn_lens
        _, sent_ids_caption, _ = self.process_sent(sent, self.max_words_in_sent)

        # print(vid_name, cap, len(cap), sent_ids_caption, len(sent_ids_caption))
        out['sent_ids_caption'] = sent_ids_caption
        if True:
            out = self.get_caption_outs(out, sent, self.ref_graphs[sent])

        return out

    def __len__(self):
        return len(self.data)

def collate_fn(batch): # batch: ( data, cap, feat)
    batch.sort(key=lambda x:len(x['sent_ids_caption']), reverse=True)
    outs = {}
    device = torch.device('cuda')
    for key in ['names', 'verb_masks', 'noun_masks', 'node_roles', 'rel_edges', 'sent_ids', 'attn_fts', 'attn_lens', 'label','sent_lens','sent_ids_caption','pos_tag','motion_fts']:
        if key in batch[0]:
            outs[key] = [x[key] for x in batch]
    
    feature, motion_fts, cap, pos_tag = outs['attn_fts'], outs['motion_fts'], outs['sent_ids_caption'], outs['pos_tag']

    max_len = len(cap[0])  # the first captions must has be the longest
    feats = torch.stack(feature, dim=0)

    motions = torch.stack(motion_fts, dim=0)
    caps = []
    pos = []
    for i in range(len(cap)):  #  for each data in the batch:
        temp_cap = [PAD] * (max_len)
        temp_cap[0:len(cap[i])] = cap[i]  # here print the original and temp_cap, for compared
        caps.append(temp_cap)
        pos_cap = [2] * (max_len)
        pos_cap[0:len(pos_tag[i])] = pos_tag[i]
        pos.append(pos_cap)
    caps = torch.LongTensor(caps)
    pos = torch.LongTensor(pos)
    outs['sent_ids_caption'] = caps.to(device)

    outs['attn_fts'] = torch.FloatTensor(feats).to(device)
    outs['motion_fts'] = torch.FloatTensor(motions).to(device)
    # reduce caption_ids lens
    outs['attn_lens'] = torch.LongTensor(outs['attn_lens']).to(device)
    outs['pos_tag'] = pos.to(device)
    if 'sent_ids' in outs:
        max_cap_len = np.max(outs['sent_lens'])
        outs['sent_ids'] = torch.LongTensor(outs['sent_ids'])[:, :max_cap_len].to(device)
        outs['sent_lens'] = torch.LongTensor(outs['sent_lens']).to(device)
        outs['verb_masks'] = torch.BoolTensor(np.array(outs['verb_masks'])[:, :, :max_cap_len]).to(device)
        outs['noun_masks'] = torch.BoolTensor(np.array(outs['noun_masks'])[:, :, :max_cap_len]).to(device)
        outs['node_roles'] = torch.LongTensor(outs['node_roles']).to(device)
        outs['rel_edges'] = torch.FloatTensor(outs['rel_edges']).to(device)
    
    #### to cuda #####
    
    # for key in ['verb_masks', 'noun_masks', 'node_roles', 'rel_edges', 'sent_ids', 'attn_fts', 'attn_lens', 'label', 'sent_lens','sent_ids_caption','pos_tag','motion_fts']:
    #     if key in batch[0]:
    #         # outs[key] =outs[key].to(device)
    #         try:
    #             print(outs[key].device)
    #         except:
    #             continue
    
    return outs

class DictDataset_test(data.Dataset):
    #用于加载train和val集
    def __init__(self, dataset, feat_name, split, vid_max_len):
        self.vid_max_len = vid_max_len
        self.dataset = dataset
        test = 'public'
        if self.dataset == 'VATEX':
            self.data = json.load(open('/mnt/hdd4/leiyu/%s/wu/data/dict_%s_data.json' % (dataset, split), 'rb'))

        # self.feat_path_c3d = '/mnt/hdd4/leiyu/all_dataset/VATEX/c3d'
        self.feat_path_i3d = '/mnt/hdd4/leiyu/VATEX/public_test'
        self.feat_path = '/mnt/hdd4/leiyu/all_dataset/VATEX/irv2' # com


        # self.feat_path = '/mnt/hdd4/leiyu/VATEX/public_test'
        # self.feat_path_c3d = '/mnt/hdd4/leiyu/VATEX/public_test'

    def __getitem__(self, index):
        data = self.data[index] #.type(torch.LongTensor)
        vid_id = data['videoID']
        name = vid_id

        cap = data['enCap']
 
        vid_id = vid_id + '.npy'


        feat = np.load(open(os.path.join(self.feat_path, vid_id), 'rb'))
        if len(feat.shape) == 1:
            feat = np.expand_dims(feat, axis=0)
        feat, _  = get_sub_frames(feat, self.vid_max_len)
        # feat_motion = np.load(open(os.path.join(self.feat_path_c3d, vid_id), 'rb'))
        motion = np.load(open(os.path.join(self.feat_path_i3d, vid_id), 'rb'))
        motion, _  = get_sub_frames(motion.squeeze(), self.vid_max_len)
        feat = torch.from_numpy(feat).float()
        # feature = feature.mean(axis=(2, 3))[None, :]
        motion = torch.from_numpy(motion).float()

        return feat, motion, cap, name 


    def __len__(self):
        return len(self.data)

def collate_fn_test(batch): # batch: ( data, cap    , feat)
    feature, motion, cap, name = zip(*batch)  # gts:
    feats = torch.stack(feature, dim=0)
    motions = torch.stack(motion, dim=0)
    caps = []
    for w in cap:  #  for each data in the batch:
        gt = []
        for v in w:
            gt.append(v)
        caps.append(gt)

    return feats, motions, caps, name # feat:(m,28,1536)