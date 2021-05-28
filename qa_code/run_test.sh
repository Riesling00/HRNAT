#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 python test.py --exp_name LSTM_POS0.1re0.4ma3.25 --dataset VATEX \
 --feat_name msrvtt_inpRes_rgb --feature_size 1024 --batch_size 512 \
 --warmup 10000 --learning_rate_decay_start 0 --scheduled_sampling_start 0 \
 --learning_rate 3e-4 