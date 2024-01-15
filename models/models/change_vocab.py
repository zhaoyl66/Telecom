import os.path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from tqdm import tqdm
import sys
# sys.path.append('需要作为模块引入的路径')
# 添加当前路径的前一级文件作为源文件夹
print("sys.path",sys.path)
path = os.path.dirname(os.path.dirname(__file__)) 
print(path)
# from .modeling_baichuan import BaiChuanForCausalLM
# from .tokenization_baichuan import BaiChuanTokenizer


class VocabularyPruner(object):

    def check(self, old_model_name_or_path, new_model_name_or_path, text):
        # 检查模型裁剪后，生成结果是否一致
        max_length = 20

        # 使用老模型对文本编码
        old_tokenizer = AutoTokenizer.from_pretrained("/data/fkj2023/pretrained_model/baichuan-7B", trust_remote_code=True,local_files_only=True)
        old_model = AutoModelForCausalLM.from_pretrained("/data/fkj2023/pretrained_model/baichuan-7B", device_map="auto", trust_remote_code=True,local_files_only=True)
        print("old_tokenizer",dir(old_tokenizer))
        # old_model = BaiChuanForCausalLM.from_pretrained(old_model_name_or_path)
        # old_tokenizer = BaiChuanTokenizer.from_pretrained(old_model_name_or_path)
        old_input_ids = old_tokenizer(text, return_tensors='pt').input_ids
        old_output = old_model.generate(old_input_ids, max_length=max_length)
        old_output_text = old_tokenizer.batch_decode(old_output)
        print('old_output:{}'.format(old_output_text))

        # 使用新模型对文本编码
        new_model = AutoModelForCausalLM.from_pretrained(new_model_name_or_path, trust_remote_code=True)
        new_tokenizer = AutoTokenizer.from_pretrained("/data/fkj2023/pretrained_model/baichuan-7B", trust_remote_code=True)
        new_input_ids = new_tokenizer(text, return_tensors='pt').input_ids
        print(new_input_ids)
        tokens = new_tokenizer.tokenize(text)
        print("TOKEN",tokens)
        new_output = new_model.generate(new_input_ids, max_length=max_length)
        new_output_text = new_tokenizer.batch_decode(new_output)
        print('new_output:{}'.format(new_output_text))

        if old_output_text == new_output_text:
            print('output is same, succeed to prune.')
        else:
            print('output is not same, fail to prune.')

    def update_ebeddings(self, model, new2old_token_id, new_embeds, new_lm_head):
        raise NotImplemented

    def prune(self, model_name_or_path, new_tokenizer_name_or_path, save_path, new_name_or_path=None):
        # 创建输出目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 加载新词表。如果是中文，就是中文的词表
        # new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_name_or_path)
        new_tokenizer = '/data/zyl2023/hpt/data/Telecom1/Telecom_vocab.txt'
        new_vocab = {}
        count = 0
        with open(new_tokenizer, 'r') as f:
            text = f.read().splitlines()
            for line in text:
                new_vocab[line] = count
                count = count + 1
        # 加载原词表。一般为多语言模型的词表
        old_tokenizer = AutoTokenizer.from_pretrained("/data/fkj2023/pretrained_model/baichuan-7B", trust_remote_code=True,local_files_only=True)
        print(old_tokenizer.vocab_files_names)
        print("old_tokenizer",dir(old_tokenizer))
        # 检查新词表是否为原词表的子集
        # old_vocab = old_tokenizer.vocab
        old_vocab = old_tokenizer.get_vocab()
        print(type(old_vocab.keys()))
        # print("new_vocab",new_vocab)
        print(type(new_vocab.keys()))
        
        count = 0
        for token in tqdm(new_vocab.keys()):
            if token not in old_vocab:
                # print('{} not exist'.format(token))
                count = count + 1
                raise Exception('{} not exist'.format(token))
        print(count)
        print('new_tokenizer is subset of old_tokenizer')

        # 获得新词表中每个token_id到原词表的token_id的映射
        new2old_token_id = {}

        not_count = 0
        for token, token_id in tqdm(new_vocab.items()):
            if token in old_vocab:
                old_token_id = old_vocab[token]
                new2old_token_id[token_id] = old_token_id
            else:
                not_count = not_count + 1
        print("not_count",not_count)
        print("new2old_token_id len",len(new2old_token_id))

        # 加载多语言模型
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True, torch_dtype='auto')
        # 计算原模型的参数量
        old_params = sum(p.numel() for p in model.parameters())
        print("Total params of original model: %.2fM" % (old_params / 1e6))

        # 对于新词表中的每个token，取出其对应的权重，复制到新模型中
        vocab_size = len(new_vocab)
        hidden_size = model.config.hidden_size
        print("vocab_size",vocab_size)
        print("hidden_size",hidden_size)
        new_embeds = torch.nn.Embedding(vocab_size, hidden_size, dtype=model.dtype)
        new_lm_head = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=False, dtype=model.dtype)
        # 更新词表权重
        self.update_ebeddings(model, new2old_token_id, new_embeds, new_lm_head)

        model.config.__dict__['vocab_size'] = vocab_size
        if new_name_or_path is not None:
            model.config.__dict__['_name_or_path'] = new_name_or_path

        # 计算新模型的参数量
        new_params = sum(p.numel() for p in model.parameters())
        print("Total params of new model : %.2fM" % (new_params / 1e6))

        print('词表缩小为原来的:{}%'.format(round(len(new_tokenizer) / len(old_tokenizer), 4)*100))
        print('模型参数量缩小为原来的:{}%'.format(round(new_params / old_params, 4)*100))
        model.save_pretrained(save_path)
        new_tokenizer.save_pretrained(save_path)


class BloomVocabularyPruner(VocabularyPruner):

    def update_ebeddings(self, model, new2old_token_id, new_embeds, new_lm_head):
        for token_id, old_token_id in tqdm(new2old_token_id.items()):
            # print("token_id",token_id)
            # print("old_token_id",old_token_id)
            # print("new_embeds.weight.data",len(new_embeds.weight.data))
            new_embeds.weight.data[token_id] = model.get_input_embeddings().weight.data[old_token_id]
            new_lm_head.weight.data[token_id] = model.lm_head.weight.data[old_token_id]
        model.set_input_embeddings(new_embeds)
        model.lm_head.weight = new_lm_head.weight

# 需要进行裁剪的模型路径
model_name_or_path = '/data/fkj2023/pretrained_model/baichuan-7B'
# 自己制作的词表的路
new_tokenizer_name_or_path = '/data/zyl2023/bert-base-chinese'
save_path = '/data/zyl2023/baichuan-7B'
pruner = BloomVocabularyPruner()
# 裁剪
pruner.prune(model_name_or_path, new_tokenizer_name_or_path, save_path)
# 检查裁剪的模型与原模型是否一致
pruner.check(model_name_or_path, save_path, text='长风破浪会有时')