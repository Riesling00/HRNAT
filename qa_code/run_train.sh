#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name pos_global_drop_0.2 --dataset MSVD \
 --feat_name msrvtt_inpRes_rgb --feature_size 1024 --batch_size 32 \
 --warmup 10000 --learning_rate_decay_start 3 --scheduled_sampling_start 3 \
 --learning_rate 3e-4 
