# Telecom
This repository contains the data, scripts and baseline codes for Multi-turn Dialogue Responsibility Assignment Task.

# Requirements
```
PyTorch==1.8.0
transformers==4.23.1
datasets
torch-geometric == 2.3.0
torch-scatter == 2.0.6
torch-sparse == 0.6.12
numpy
tqdm
```

# Data
Our own dataset for segmentation is under ./data directory

# Train

```
usage: train.py [-h] [--lr LR] [--data DATA] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] --name NAME [--update UPDATE] [--model MODEL] [--wandb] [--arch ARCH] [--layer LAYER] [--graph GRAPH] [--prompt-loss] [--low-res] [--seed SEED] [--max_seq_length MAX_SEQUENCE_LENGTH] [--max_segment_num MAX_SEGMENT_NUMBER]

optional arguments:
  -h, --help                show this help message and exit
  --lr LR					Learning rate. Default: 3e-5.
  --data {Telecom}          Dataset.
  --batch BATCH             Batch size.
  --early-stop EARLY_STOP   Epoch before early stop.
  --device DEVICE           cuda or cpu. Default: cuda.
  --name NAME               A name for different runs.
  --update UPDATE           Gradient accumulate steps.
  --wandb                   Use wandb for logging.
  --seed SEED               Random seed.
  --max_seq_length          The maximum total input sequence length after WordPiece tokenization.Sequences longer than this will be truncated, and sequences shorter than this will be padded.
  --command                 define the name of the running process
  --max_segment_num         The maximum total dialogue segment number.
```
