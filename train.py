import random
import json
from transformers import AutoTokenizer,BertConfig
# from models.configuration_bert import BertConfig
from models.tokenization_bert import BertTokenizer
from transformers import BertTokenizer as Tokenizer
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import os
import datasets
from tqdm import tqdm
import argparse
import wandb
from utils_TSbert import convert_cs_to_feature

from eval import evaluate

import utils

import sys
import os
from setproctitle import setproctitle


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--data', type=str, default='Telecom')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--early-stop', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--model', type=str, default='prompt')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument("--arch", default="/data/zyl2023/bert-base-chinese", type=str, help="pretrained bert model path")
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT') #GCN
    parser.add_argument('--low-res', default=False, action='store_true')
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument("--max_segment_num",
                    default=6, #
                    type=int,
                    help="The maximum total segment number.")
    parser.add_argument("--max_seq_length",
                    default=350, #
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
    parser.add_argument("--command",type=str,default='Telecom')
    return parser



class Save:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    setproctitle(args.command)
    if args.wandb:
        wandb.init(config=args, project='HPT')
    print(args)
    utils.seed_torch(args.seed)
    config = BertConfig.from_pretrained(r'/data/zyl2023/bert-base-chinese',local_files_only=True)
    tokenizer = Tokenizer.from_pretrained(r'/data/zyl2023/bert-base-chinese',local_files_only=True)
    data_path = os.path.join('data', args.data)
    args.name = args.data + '-' + args.name
    batch_size = args.batch

    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
    label_dict = {i: v for i, v in label_dict.items()}
    print('label_dict:',label_dict)
    '''
    label_dict: 
    '''

    slot2value = torch.load(os.path.join(data_path, 'slot.pt'))
    '''
    slot2value: 
    '''
    value2slot = {}
    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1
    
    path_list = [(i, v) for v, i in value2slot.items()]
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1

    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth

    
    depth_dict = {i: get_depth(i) for i in range(num_class)}
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1

    def get_keys_by_value(d, value):
        keys = []
        for k, v in d.items():
            if v == value:
                keys.append(k)
        return keys

    keys_with_value = get_keys_by_value(depth_dict, max_depth)
    
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
    

    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))

    if args.model == 'prompt':
        if os.path.exists(os.path.join(data_path, args.model)):
            dataset = datasets.load_from_disk(os.path.join(data_path, args.model))
        else:
            dataset = datasets.load_dataset('json',
                                            data_files={'train': 'data/{}/{}_trainseg1_clear.json'.format(args.data, args.data),
                                                        'dev': 'data/{}/{}_testseg1_clear.json'.format(args.data, args.data),
                                                        'test': 'data/{}/{}_devseg1_clear.json'.format(args.data, args.data), })
            
            prefix = []
            for i in range(max_depth):
                prefix.append(tokenizer.vocab_size + num_class + i)
                prefix.append(tokenizer.vocab_size + num_class + max_depth)
            prefix.append(tokenizer.sep_token_id)

            def data_map_function(batch, tokenizer): ## embedding
                new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': [],'seg_input_ids':[],'seg_token_type_ids':[],'seg_attention_mask':[],'cls_sep_pos':[],'true_len':[]}
                for l, t in zip(batch['label'], batch['token']):
                    new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)])
                    for d in range(max_depth):
                        for i in depth2label[d]:
                            new_batch['labels'][-1][d][i] = 0
                        for i in l:
                            if new_batch['labels'][-1][d][i] == 0:
                                new_batch['labels'][-1][d][i] = 1
                    new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y]

                    
                    lines = t.split('\t')
                    context = lines[:-1]
                    response = lines[-1]
                    
                    feature = convert_cs_to_feature(prefix,context,response,max_seg_num=args.max_segment_num,max_seq_length=args.max_seq_length - len(prefix), tokenizer=tokenizer)
                    # new_batch['seg_input_ids'].append(feature['seg_input_ids'])
                    # new_batch['seg_token_type_ids'].append(feature['seg_token_type_ids'])
                    # new_batch['seg_attention_mask'].append(feature['seg_attention_mask'])
                    new_batch['cls_sep_pos'].append(feature['cls_sep_pos'])
                    new_batch['true_len'].append(feature['true_len'])

                    new_batch['input_ids'].append(feature['seg_input_ids'])
                    new_batch['attention_mask'].append(feature['seg_attention_mask'])
                    new_batch['token_type_ids'].append([0] * args.max_seq_length)
                    
                return new_batch


            
            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            dataset.save_to_disk(os.path.join(data_path, args.model))
        dataset['train'].set_format('torch', columns=['cls_sep_pos','true_len', 'labels','input_ids','attention_mask','token_type_ids'])
        dataset['dev'].set_format('torch', columns=['cls_sep_pos','true_len', 'labels','input_ids','attention_mask','token_type_ids'])
        dataset['test'].set_format('torch', columns=['cls_sep_pos','true_len', 'labels','input_ids','attention_mask','token_type_ids'])

        from models.prompt import Prompt

    else:
        raise NotImplementedError
    if args.low_res:
        if os.path.exists(os.path.join(data_path, 'low.json')):
            index = json.load(open(os.path.join(data_path, 'low.json'), 'r'))
        else:
            index = [i for i in range(len(dataset['train']))]
            random.shuffle(index)
            json.dump(index, open(os.path.join(data_path, 'low.json'), 'w'))
        dataset['train'] = dataset['train'].select(index[len(index) // 5:len(index) // 10 * 3])
    print('len(label_dict):',len(label_dict))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Prompt.from_pretrained(args.arch,config=config,tokenizer=tokenizer ,max_depth = max_depth, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                   graph_type=args.graph, data_path=data_path, depth2label=depth2label,max_seg_num=args.max_segment_num,
                                   max_seq_len=args.max_seq_length, device=device, local_files_only=True)
    model.init_embedding() 

    model.to('cuda')
    if args.wandb:
        wandb.watch(model)

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    dev = DataLoader(dataset['dev'], batch_size=8, shuffle=False)
    model.to('cuda')

    optimizer = Adam(model.parameters(), lr=args.lr)

    save = Save(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    loss = 0
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))

    for epoch in range(1000):
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break

        model.train()  ##train
        with tqdm(train) as p_bar:
            for batch in p_bar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = model(**batch)
                output['loss'].backward()
                loss += output['loss'].item()
                if args.wandb:
                    wandb.log({'loss': loss, })
                p_bar.set_description(
                    'loss:{:.4f}'.format(loss, ))
                optimizer.step()
                optimizer.zero_grad()
                loss = 0

        model.eval()
        pred = []
        gold = []
        with torch.no_grad(), tqdm(dev) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids, logits = model.generate(input_ids=batch['input_ids'],token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'],seg_input_ids=batch['seg_input_ids'],cls_sep_pos=batch['cls_sep_pos'], seg_token_type_ids=batch['seg_token_type_ids'], seg_attention_mask=batch['seg_attention_mask'], true_len=batch['true_len'], depth2label=depth2label, )
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        precision = scores['precision']
        recall = scores['recall']
        print('F 1******')
        print('macro', macro_f1, 'micro', micro_f1, 'precision',precision,'recall',recall)
        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1})
        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0
        # save(macro_f1, best_score, os.path.join('checkpoints', args.name, 'checkpoint_{:d}.pt'.format(epoch)))
        save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
        if args.wandb:
            wandb.log({'best_macro': best_score_macro, 'best_micro': best_score_micro})

        torch.cuda.empty_cache()

    # test
    test = DataLoader(dataset['test'], batch_size=8, shuffle=False)
    model.eval()

    def test_function(extra):
        checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(extra)),
                                map_location='cpu')
        model.load_state_dict(checkpoint['param'])
        pred = []
        gold = []
        with torch.no_grad(), tqdm(test) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids, logits = model.generate(input_ids = batch['input_ids'],attention_mask=torch.tensor(batch['attention_mask']),token_type_ids=batch['token_type_ids'],seg_input_ids=batch['seg_input_ids'],cls_sep_pos=batch['cls_sep_pos'], seg_token_type_ids=batch['seg_token_type_ids'], seg_attention_mask=batch['seg_attention_mask'], true_len=batch['true_len'], depth2label=depth2label, )
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        precision = scores['precision']
        recall = scores['recall']
        print("test result:")
        print('macro', macro_f1, 'micro', micro_f1, 'precision',precision,'recall',recall)
        with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
            print('macro', macro_f1, 'micro', micro_f1, file=f)
            prefix = 'test' + extra
        if args.wandb:
            wandb.log({prefix + '_macro': macro_f1, prefix + '_micro': micro_f1})


    # test_function('_macro')
    test_function('_micro')
