#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python train.py --exp_name hier_i3d+irv2_pos0.25 --dataset VATEX \
 --feat_name msrvtt_inpRes_rgb --feature_size 1024 --batch_size 64 \
 --warmup 10000 --learning_rate_decay_start 3 --scheduled_sampling_start 3 \
 --learning_rate 3e-4 
