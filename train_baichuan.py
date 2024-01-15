import random
import json
from transformers import AutoTokenizer, BertConfig, BertTokenizer
import torch
from torch.utils.data import Subset, DataLoader
from models.configuration_utils import PretrainedConfig
from models.tokenization_baichuan import BaiChuanTokenizer
from torch.optim import Adam
import torch.nn as nn
import os
import datasets
from tqdm import tqdm
import argparse
import wandb

from eval import evaluate,eval_gt_2,evaluate_layer1,evaluate_layer2

import utils
import os
import sys


from setproctitle import setproctitle

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--data', type=str, default='WebOfScience')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--early-stop', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--model', type=str, default='prompt')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument("--arch", default="/data/zyl2023/bert-base-chinese", type=str, help="pretrained bert model path")
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT')
    parser.add_argument('--low-res', default=False, action='store_true')
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument("--cuda",type=str,default='3')
    parser.add_argument("--command",type=str,default='hpt')
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,3"   #GPU设备号

    if args.wandb:
        wandb.init(config=args, project='HPT')
    utils.seed_torch(args.seed)
    tokenizer = BaiChuanTokenizer.from_pretrained(r'/data/zyl2023/baichuan-7B',trust_remote_code = True, local_files_only=True)
    config = PretrainedConfig.from_pretrained(r'/data/zyl2023/bert-base-chinese',trust_remote_code=True,local_files_only=True)

    data_path = os.path.join('data', args.data)
    args.name = args.data + '-' + args.name
    batch_size = args.batch

    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
    label_dict = {i: v for i, v in label_dict.items()}


    slot2value = torch.load(os.path.join(data_path, 'slot.pt'))

    value2slot = {}
    num_class = 0
    for s in slot2value: 
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1                  
    print('num_class',num_class)                            #label总数
    # print("value2slot",value2slot)
    path_list = [(i, v) for v, i in value2slot.items()]
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1
    # print('path_list:',path_list)

    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth


    depth_dict = {i: get_depth(i) for i in range(num_class)}
    # print('depth_dict: ',depth_dict)
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
    print('max_depth:',max_depth)
    # print('depth2label:',depth2label)

    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))

    if args.model == 'prompt':
        if os.path.exists(os.path.join(data_path, args.model)):
            dataset = datasets.load_from_disk(os.path.join(data_path, args.model))
        else:                                       'test': 'data/{}/{}_test.json'.format(args.data, args.data), })
            dataset = datasets.load_dataset('json',
                                            data_files={'train': 'data/{}/{}_trainseg1_clear.json'.format(args.data, args.data),
                                                        'dev': 'data/{}/{}_devseg1_clear.json'.format(args.data, args.data),
                                                        'test': 'data/{}/{}_testseg1_clear.json'.format(args.data, args.data), })

            prefix = []
            for i in range(max_depth):
                prefix.append(tokenizer.vocab_size + num_class + i)
                prefix.append(tokenizer.vocab_size + num_class + max_depth)
            prefix.append(tokenizer.eos_token_id)
            print("tokenizer.eos_token_id",tokenizer.eos_token_id)
            print("tokenizer.bos_token_id",tokenizer.bos_token_id)


            def data_map_function(batch, tokenizer): ## embedding
                new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
                for l, t in zip(batch['label'], batch['token']):
                    new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)])
                    for d in range(max_depth):
                        for i in depth2label[d]:
                            new_batch['labels'][-1][d][i] = 0
                        for i in l:
                            if new_batch['labels'][-1][d][i] == 0:
                                new_batch['labels'][-1][d][i] = 1
                    new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y]

                    tokens = tokenizer(t, truncation=True)
                    new_batch['input_ids'].append([tokenizer.bos_token_id] + tokens['input_ids'][:-1][-348 + len(prefix):] 
                                                  + [tokenizer.eos_token_id] + prefix)
                    new_batch['input_ids'][-1].extend(
                        [tokenizer.pad_token_id] * (350 - len(new_batch['input_ids'][-1])))
                    new_batch['attention_mask'].append(
                        [1] + tokens['attention_mask'][:-1][-348 + len(prefix):] + [1] + [1] * len(prefix))
                    new_batch['attention_mask'][-1].extend([0] * (350 - len(new_batch['attention_mask'][-1])))
                    new_batch['token_type_ids'].append([0] * 350)
                return new_batch

            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            dataset.save_to_disk(os.path.join(data_path, args.model))
        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])

        from models.prompt_baichuan import Prompt

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
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    model = Prompt.from_pretrained(args.arch,config=config,tokenizer=tokenizer,num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                   graph_type=args.graph, data_path=data_path, depth2label=depth2label, device=device)
    model.init_embedding()          #embedding
    from accelerate import dispatch_model

    from accelerate import infer_auto_device_map,load_checkpoint_in_model

    model.to(device)
    if args.wandb:
        wandb.watch(model)

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, )
    dev = DataLoader(dataset['dev'], batch_size=8, shuffle=False)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    model, optimizer, train, dev = accelerator.prepare(model, optimizer, train, dev)

    save = Save(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    update_step = 0
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
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = model(**batch)
                output['loss'].backward()
                accelerator.backward(output['loss'])
                update_step += 1
                if update_step % args.update == 0:
                    if args.wandb:
                        wandb.log({'loss': loss, })
                    p_bar.set_description(
                        'loss:{:.4f}'.format(loss, ))
                    optimizer.step()
                    optimizer.zero_grad()
                    loss = 0
                    update_step = 0
         

        model.eval()
        pred = []
        gold = []
        with torch.no_grad(), tqdm(dev) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                print("batch['input_ids'].device",batch['input_ids'].device)
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, )
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
        recall = scores['precision']
        print('macro', macro_f1, 'micro', micro_f1,'precision',precision,'recall',recall)
        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1})
        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            # save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            accelerator.wait_for_everyone()
            # 只在主进程中保存模型
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model,os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            # save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            accelerator.wait_for_everyone()
            # 只在主进程中保存模型
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model,os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0
        # save(macro_f1, best_score, os.path.join('checkpoints', args.name, 'checkpoint_{:d}.pt'.format(epoch)))
        
        save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))

        # 等待所有进程达到一定程度后再执行指令
        accelerator.wait_for_everyone()
        # 只在主进程中保存模型
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model,os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
            # torch.save(unwrapped_model.state_dict, "./model/accelerate.pt") 

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
                batch = {k: v.to('cuda') for k, v in batch.items()}
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, )
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
        scores = evaluate(pred, gold, label_dict)
        gt_1 = eval_gt_2(pred, gold,)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        precision = scores['precision']
        recall = scores['precision']
        layer_1 = evaluate_layer1(pred, gold,label_dict)
        layer_2 = evaluate_layer2(pred, gold,label_dict)
        print("text result:")
        print('macro', macro_f1, 'micro', micro_f1, 'precision',precision,'recall',recall,'layer1',layer_1,'layer2',layer_2)
        
        with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
            print('macro', macro_f1, 'micro', micro_f1, file=f)
            prefix = 'test' + extra
        if args.wandb:
            wandb.log({prefix + '_macro': macro_f1, prefix + '_micro': micro_f1})


    # test_function('_macro')
    test_function('_micro')
