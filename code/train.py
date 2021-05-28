import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
from data_io import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from utils.utils import Logger
import sys
sys.path.append('/mnt/hdd4/leiyu/workspace/video')
import evaluation
from evaluation import PTBTokenizer, Cider
from models import LSTMCaptionModel
import torch
import json
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing 
from shutil import copyfile
# torch.multiprocessing.set_start_method('spawn')
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

class DataLoaderX(DataLoader):

    def __iter__(self):
        
        return BackgroundGenerator(super().__iter__())

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

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (feats, captions) in enumerate(dataloader):
                feats, captions = feats.to(device), captions.to(
                    device)
                out = model(feats, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, word_to_id, idxs_word):
    import itertools
    model.eval()
    gen = {} 
    save_out= {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (feats, motions, caps_gt, vid_id) in enumerate(iter(dataloader)):
            feats = feats.to(device)
            motions = motions.to(device)
            with torch.no_grad():
                out, _ = model.beam_search((feats,motions), 26, word_to_id['<EOS>'], 5, out_size=1)
            caps_gen = decode(out, idxs_word, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                d = {vid_id[i]:gen_i}
                save_out.update(d)   
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    # fi = open("/home/leiyu/workspace/video_ma+de/saved_models/en_captions_%s_test.json" % args.exp_name,'w')
    # json.dump(save_out,fi)
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(args, model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    # scheduler.step()
    running_loss = .0
    running_loss_1 = .0
    running_loss_2 = .0
    running_loss_3 = .0
    running_loss_4 = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (batch_data) in enumerate(dataloader):
            captions = batch_data['sent_ids_caption']
            out, loss_match, loss_re, loss_pos = model(args, batch_data)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()

            loss = args.xe_lambda * loss_fn(out.view(-1, len(text_field)), captions_gt.view(-1))
            running_loss_1 += loss.item()
            
            if args.matching_flag is True:
                loss += loss_match  
                running_loss_2 += loss_match.item()

            if args.reconstructor_flag is True:
                loss += loss_re
                running_loss_3 += loss_re.item()

            if args.pos_flag is True and args.backpos is True:
                loss += loss_pos
                running_loss_4 += loss_pos.item()

            loss.backward()
            if args.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                # torch.nn.utils.clip_grad_value_(model.parameters(), number)

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1), loss_main = running_loss_1/ (it + 1), loss_mat = running_loss_2/(it+1), loss_re = running_loss_3/(it+1), loss_postag = running_loss_4/(it+1))
            pbar.update()
            # scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='VATEX TRAINING')
        self.parser.add_argument('--exp_name', type=str, default='LSTM')
        self.parser.add_argument('--dataset', type=str, default='VATEX')
        self.parser.add_argument('--feat_name', type=str, default='msrvtt_inpRes_rgb')
        self.parser.add_argument('--batch_size', type=int, default=5) 
        self.parser.add_argument('--vocab_size', type=int, default=10424)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument('--word_embed', type=int, default=300)
        self.parser.add_argument('--rnn_size', type=int, default=512)
        self.parser.add_argument('--att_size', type=int, default=512)
        self.parser.add_argument('--xe_lambda', type=float, default=1.0)
        self.parser.add_argument('--recon_lambda', type=float, default=0.2) 
        self.parser.add_argument('--pf_lambda', type=float, default=0.7)
        self.parser.add_argument('--cls_lambda', type=float, default=0.8)
        self.parser.add_argument('--matching_flag', type=bool, default=False)
        self.parser.add_argument('--pos_flag', type=bool, default=True)
        self.parser.add_argument('--backpos', type=bool, default=True)      
        self.parser.add_argument('--pos_lambda', type=float, default=0.25)           
        self.parser.add_argument('--matching_txt_load', type=bool, default=False)
        self.parser.add_argument('--ss_depth', type=bool, default=False)
        self.parser.add_argument('--fixed', type=bool, default=False)
        self.parser.add_argument('--fine_tune', type=bool, default=False)
        self.parser.add_argument('--num_verbs', type = int, default=4)
        self.parser.add_argument('--num_nouns', type = int, default=6)
        self.parser.add_argument('--vse_loss_weight', type=float, default=3.0)
        self.parser.add_argument('--ref_graph_file', type = str, default='/mnt/hdd4/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/sent2rolegraph.augment.json') 
        self.parser.add_argument('--reconstructor_flag', type=bool, default=False)
        self.parser.add_argument('--predict_forward_flag', type=bool, default=False)
        self.parser.add_argument('--action_recognition', type=bool, default=False)
        self.parser.add_argument('--max_words_in_sent', type=int, default=30)
        self.parser.add_argument('--vid_max_len', type=int, default=26)
        self.parser.add_argument('--gradient_clip', type=int, default=2)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
        self.parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                            help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
        self.parser.add_argument('--learning_rate_decay_every', type=int, default=4,
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
        self.parser.add_argument('--workers', type=int, default=1)
        self.parser.add_argument('--warmup', type=int, default=10000)
        self.parser.add_argument('--resume_last', default = False, action='store_true')
        self.parser.add_argument('--resume_best', default = True, action='store_true')
        self.parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
        self.args = self.parser.parse_args()
 

if __name__ == '__main__':
    device = torch.device('cuda')
    param = Param()
    args = param.args

    logger = Logger(os.path.join('saved_models', '%s_log.txt'%(args.exp_name)))

    print(args)
    logger.write('LSTM Training')
    torch.multiprocessing.set_start_method('spawn')
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # vocab = json.load(open('/mnt/hdd1/leiyu/%s/wu/data/vocab.json' %(args.dataset),'rb'))
    # word_to_id = vocab['wti']
    # id_to_word = vocab['itw']
    word_to_id = json.load(open('/mnt/hdd4/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/word2int.json', 'rb'))
    id_to_word = json.load(open('/mnt/hdd4/leiyu/all_dataset/HGR_T2V/VATEX/annotation/RET/int2word.json', 'rb'))
    # Create the dataset
    train_dataset = TrainValDataset(args, args.dataset, args.feat_name, 'train', args.vid_max_len)

    # Model and dataloaders
    model = LSTMCaptionModel(args, 30, len(word_to_id), word_to_id['<BOS>']).to(device)
    
    # ================ embedding ====================#
    # checkpoint_embed = torch.load(dir_embed)
    # print('loading the best embedding model --------')
    # model.state_dict()['embed.weight']=checkpoint_embed['text_encoder']['embedding.we.weight']

    # dict_dataset_train = DictDataset(args.dataset, args.feat_name, 'train', 30)
    # dict_dataset_val = DictDataset(args.dataset, args.feat_name, 'val', 30)
    dict_dataset_val = DictDataset_test(args.dataset, args.feat_name, 'test', args.vid_max_len)

    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.rnn_size ** -.5) * min(s ** -.5, s * warm_up ** -1.5)


    # Initial conditions
    # optim = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    optim = Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98))
    # scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=word_to_id['<PAD>'])
    use_rl = False
    best_cider = .0
    patience = 0
    start_epoch = 0

    if args.fine_tune is True:
        fname = '/mnt/hdd1/leiyu/workspace/video_re_ma/saved_models/LSTM_1_best.pth'
        print('loading the best model --------')
        checkpoint = torch.load(fname)
        if args.matching_txt_load:
            checkpoint_ret = torch.load(dir_embed)
            for key, value in checkpoint_ret['text_encoder'].items():
                key = 'txt_embed.' + key
                checkpoint['state_dict'][key] = value
        try:
            checkpoint.train()
        except AttributeError as error:
            print('error')
        
        model.load_state_dict(checkpoint['state_dict'])

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            # scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            logger.write('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    logger.write("Training starts")
    for e in range(start_epoch, start_epoch + 40):
        if  e > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
            frac = int((e - args.learning_rate_decay_start) / args.learning_rate_decay_every)
            decay_factor = args.learning_rate_decay_rate ** frac
            args.current_lr = args.learning_rate * decay_factor
            set_lr(optim, args.current_lr)  # set the decayed rate
        # print('epoch {}, lr_decay_start {}, cur_lr {}'.format(epoch, opt.learning_rate_decay_start, opt.current_lr))
        else:
            args.current_lr = args.learning_rate
        # Assign the scheduled sampling prob
        if  e > args.scheduled_sampling_start and args.scheduled_sampling_start >= 0:
            frac = int((e - args.scheduled_sampling_start) / args.scheduled_sampling_increase_every)
            args.ss_prob = min(args.scheduled_sampling_increase_prob * frac, args.scheduled_sampling_max_prob)
            model.ss_prob = args.ss_prob

        
        logger.write('epoch %d - learning rate: %f' % (e, optim.param_groups[0]['lr']))
        logger.write('args.current_lr :%f' % (args.current_lr))
        # dataloader_train = DataLoaderX(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=args.workers,
        #                               drop_last=True)
        dataloader_train = DataLoaderX(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=args.workers,
                                       drop_last=True)
        # dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=args.workers)
        # dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, collate_fn=collate_fn_dict, shuffle=True,
        #                                    num_workers=args.workers)
 
        dict_dataloader_val = DataLoaderX(dict_dataset_val, collate_fn=collate_fn_test, batch_size=args.batch_size // 5)
        # dict_dataloader_val = DataLoaderX(dict_dataset_val, collate_fn=collate_fn_test, batch_size=args.batch_size // 5)
               
        # if not use_rl:
        train_loss = train_xe(args, model, dataloader_train, optim, word_to_id)
        logger.write('epoch %d - train, loss: %.2f' % (e, train_loss))
        writer.add_scalar('data/train_loss', train_loss, e)

        val_loss =0.

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, word_to_id, id_to_word)
        val_cider = scores['CIDEr']
        logger.write("Epoch %d - val, cider: %.4f, bleu 0: %.4f, bleu 4: %.4f, meteor: %.4f, rough: %.4f"
                     % (e, scores['CIDEr'], scores['BLEU'][0], scores['BLEU'][3], scores['METEOR'], scores['ROUGE']))
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)


        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        # else:
        #     patience += 1

        # switch_to_rl = False
        exit_train = False
        if patience == 5:
            # if not use_rl:
            #     use_rl = True
            #     switch_to_rl = True
            #     patience = 0
            #     optim = Adam(model.parameters(), lr=5e-6)
            #     logger.write("Switching to RL")
            # else:
            logger.write('patience reached.')
            exit_train = True

        if not best:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            logger.write('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        # if switch_to_rl and not best:
        #     data = torch.load('saved_models/%s_best.pth' % args.exp_name)
        #     torch.set_rng_state(data['torch_rng_state'])
        #     torch.cuda.set_rng_state(data['cuda_rng_state'])
        #     np.random.set_state(data['numpy_rng_state'])
        #     random.setstate(data['random_rng_state'])
        #     model.load_state_dict(data['state_dict'])
        #     logger.write('Resuming from epoch %d, validation loss %f, and best cider %f' % (
        #         data['epoch'], data['val_loss'], data['best_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)

        if best:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)

        if exit_train:
            writer.close()
            break
