import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
from data_io import *
from torch.utils.data import DataLoader
from utils.utils import Logger
import sys
sys.path.append('/home/leiyu/workspace/video')
import evaluation
from evaluation import PTBTokenizer, Cider
from models import LSTMCaptionModel
import torch
import json
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def decode(word_idxs, idxs_word, join_words=True):
    captions = []
    # print(idxs_word)
    for wis in word_idxs:
        caption = []
        for wi in wis:
            word = idxs_word[str(int(wi))]
            if word == '<EOS>':
                break
            caption.append(word)
        if join_words:
            caption = ' '.join(caption)
        captions.append(caption)
    return captions



def evaluate_metrics(model, dataloader, word_to_id, idxs_word):
    import itertools
    model.eval()
    save_out= {}    
    gen = {}
    gts = {}
    with tqdm(desc=' - evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (feats, caps_gt, vid_id) in enumerate(iter(dataloader)):
            feats = feats.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(feats, 26, word_to_id['<EOS>'], 5, out_size=1)
            caps_gen = decode(out, idxs_word, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                d = {vid_id[i]:gen_i}
                save_out.update(d)   
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()
    # fi = open("/home/leiyu/workspace/en_captions_%s_test.json" % args.exp_name,'w')
    # json.dump(save_out,fi)
    gts = evaluation.PTBTokenizer.tokenize(gts) 
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='VATEX TRAINING')
        self.parser.add_argument('--exp_name', type=str, default='LSTM')
        self.parser.add_argument('--dataset', type=str, default='VATEX')
        self.parser.add_argument('--feat_name', type=str, default='msrvtt_inpRes_rgb')
        self.parser.add_argument('--batch_size', type=int, default=5)
        self.parser.add_argument('--vocab_size', type=int, default=10424)
        self.parser.add_argument('--feature_size', type=int, default=1024)
        self.parser.add_argument('--word_embed', type=int, default=300)
        self.parser.add_argument('--rnn_size', type=int, default=512)
        self.parser.add_argument('--att_size', type=int, default=512)
        self.parser.add_argument('--xe_lambda', type=float, default=1.0)
        self.parser.add_argument('--recon_lambda', type=float, default=0.4)
        self.parser.add_argument('--pf_lambda', type=float, default=0.7)
        self.parser.add_argument('--cls_lambda', type=float, default=0.8)
        self.parser.add_argument('--matching_flag', type=bool, default=True)
        self.parser.add_argument('--pos_flag', type=bool, default=True)     
        self.parser.add_argument('--pos_lambda', type=float, default=0.10)           
        self.parser.add_argument('--matching_txt_load', type=bool, default=True)
        self.parser.add_argument('--ss_depth', type=bool, default=False)
        self.parser.add_argument('--fixed', type=bool, default=True)
        self.parser.add_argument('--fine_tune', type=bool, default=False)
        self.parser.add_argument('--num_verbs', type = int, default=4)
        self.parser.add_argument('--num_nouns', type = int, default=6)
        self.parser.add_argument('--vse_loss_weight', type=float, default=3.25)
        self.parser.add_argument('--ref_graph_file', type = str, default='/mnt/hdd1/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/sent2rolegraph.augment.json') 
        self.parser.add_argument('--reconstructor_flag', type=bool, default=True)
        self.parser.add_argument('--predict_forward_flag', type=bool, default=False)
        self.parser.add_argument('--action_recognition', type=bool, default=False)
        self.parser.add_argument('--max_words_in_sent', type=int, default=30)
        self.parser.add_argument('--vid_max_len', type=int, default=30)
        self.parser.add_argument('--gradient_clip', type=int, default=10)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
        self.parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                            help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
        self.parser.add_argument('--learning_rate_decay_every', type=int, default=5,
                            help='every how many iterations thereafter to drop LR?(in epoch)')
        self.parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5,
                            help='every how many iterations thereafter to drop LR?(in epoch)')
        self.parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                            help='at what iteration to start decay gt probability')
        self.parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                            help='every how many iterations thereafter to gt probability')
        self.parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                            help='How much to update the prob')
        self.parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                            help='Maximum scheduled sampling prob.')
        self.parser.add_argument('--workers', type=int, default=0)
        self.parser.add_argument('--warmup', type=int, default=10000)
        self.parser.add_argument('--resume_last', default = False, action='store_true')
        self.parser.add_argument('--resume_best', default = False, action='store_true')
        self.parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
        self.args = self.parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda')
    param = Param()
    args = param.args
    logger = Logger(os.path.join('saved_models', '%s_log_test.txt'%(args.exp_name)))

    print(args)
    logger.write('LSTM Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Create the dataset
    # val_dataset = TrainValDataset(args.dataset, args.feat_name, 'val', 30)

    # vocab = json.load(open('/hhd3/leiyu/%s/wu/data/vocab.json' %(args.dataset),'rb'))
    # vocab = json.load(open('/mnt/hdd1/leiyu/VATEX/wu/data/vocab.json','rb'))
    # word_to_id = vocab['wti']
    # id_to_word = vocab['itw']
    word_to_id = json.load(open('/mnt/hdd1/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/word2int.json', 'rb'))
    id_to_word = json.load(open('/mnt/hdd1/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/int2word.json', 'rb'))

    # Model and dataloaders
    model = LSTMCaptionModel(args, 26, len(word_to_id), word_to_id['<BOS>']).to(device)

    dict_dataset_val = DictDataset(args.dataset, args.feat_name, 'val', args.vid_max_len)
    # dict_dataset_val = DictDataset_test(args.dataset, args.feat_name, 'val', args.vid_max_len)



    fname = 'saved_models/%s_best.pth' % args.exp_name
    checkpoint = torch.load(fname)
    try:
        checkpoint.eval()
    except AttributeError as error:
        print('error')
    model.load_state_dict(checkpoint['state_dict'])
    ### now you can evaluate it
    model.eval()

    dict_dataloader_val = DataLoader(dict_dataset_val, collate_fn=collate_fn_dict, batch_size=args.batch_size // 5)

    scores = evaluate_metrics(model, dict_dataloader_val, word_to_id, id_to_word)
    val_cider = scores['CIDEr']
    logger.write("test, cider: %.4f, bleu 0: %.4f, bleu 4: %.4f, meteor: %.4f, rough: %.4f"
                    % ( scores['CIDEr'], scores['BLEU'][0], scores['BLEU'][3], scores['METEOR'], scores['ROUGE']))


    
    writer.close()

