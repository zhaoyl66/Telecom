#!/bin/bash

nohup python -u train.py  --update 1 --max_segment_num 6  --max_seq_length 350 --early-stop 10 --data Telecom1 --batch 8 --name results$(date +%Y%m%d_%H%M) --cuda '0'  --command 'train'>>./log/train_$(date +%Y%m%d_%H%M).log 2>&1 & echo $(date +%Y%m%d_%H:%M:%S) $! >> ./log/pidfile.txt