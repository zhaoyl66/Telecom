#!/bin/bash

nohup accelerate launch --config_file default_config.yaml train_baichuan.py --name results$(date +%Y%m%d_%H%M)  --batch 2 --data Telecom1 >>./log/baichuan_emb$(date +%Y%m%d_%H:%M:%S).log 2>&1 & echo $(date +%Y%m%d_%H:%M:%S) $! >> ./log/pidfile.txt