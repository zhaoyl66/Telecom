import torch

# model = torch.load('tokenizer.model')
# print(type(model))

# with open('tokenizer.model', 'r', encoding='ISO-8859-1') as f:
#     lines = f.readlines()
#     print(lines)

from tokenization_baichuan import BaiChuanTokenizer

# Load the tokenizer
tokenizer = BaiChuanTokenizer.from_pretrained("/data/fkj2023/pretrained_model/baichuan-7B")

# Now you can use the tokenizer to encode, decode, etc.

vocabulary = list(tokenizer.get_vocab().keys())
print(len(vocabulary))
print(vocabulary[10000:10010])
# print(dir(tokenizer))
print(tokenizer.vocab_file)
print(tokenizer.vocab_files_names)
print(type(tokenizer.pretrained_vocab_files_map))
print(tokenizer.pretrained_vocab_files_map.keys())
# print(tokenizer.vocab)
