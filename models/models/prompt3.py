
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss, MSELoss
import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.autograd import Variable
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self,num_labels):
        super(MyModel,self).__init__()
        self.num_labels = num_labels
        self.affine_out = nn.Linear(in_features=4*768, out_features=2*num_labels)  #3:max_depth

    def forward(self,inputs=None,labels=None,):
        L_seg_key = self.affine_out(inputs)
        multiclass_logits = torch.softmax(L_seg_key,dim=1).squeeze(dim=-1) #batch_size, max_depth*num_label
        logits = multiclass_logits.view(-1, self.num_labels)

        return logits

def multilabel_categorical_crossentropy(y_true, y_pred):    
    loss_mask = y_true != -100
    y_true = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1))
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_true.size(-1))
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()
    
label_dict = torch.load('/data/zyl2023/hpt/data/Tele/value_dict.pt')
num_labels = len(label_dict)
model = MyModel(num_labels)
optimizer = torch.optim.Adam(model.parameters(),lr=2e-3)

model.to('cuda')

batch = torch.load('../batch.pt')
print("batch size", batch.size())
labels = torch.load('../labels.pt')
print("labels size", labels.size())

for epoch in range(10):
    model.train()
    logits = model(batch,labels)
    optimizer.zero_grad()
    if labels is not None:
        multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, num_labels), logits)
    multiclass_loss.backward()
    optimizer.step()
    loss = multiclass_loss.item()
    print("loss:",loss)

    # model.eval()
    # pred = []
    # gold = []
    
    #         batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    #         loss, logits = model.forward(inputs=batch['inputs'],labels=batch['labels'])
    #         logits = logits.view(-1, num_labels)
    #         prediction_scores = logits.view(-1,len(depth2label),logits.size(-1))
    #         predict_labels = []
    #         for scores in prediction_scores:
    #             predict_labels.append([])
    #             for i, score in enumerate(scores):
    #                 for l in depth2label[i]:
    #                     if score[l] > 0:
    #                         predict_labels[-1].append(l)

    #         for out, g in zip(predict_labels, batch['labels']):
    #             pred.append(set([i for i in out]))
    #             gold.append([])
    #             g = g.view(-1, num_labels)
    #             for ll in g:
    #                 for i, l in enumerate(ll):
    #                     if l == 1:
    #                         gold[-1].append(i)
    # scores = evaluate(pred, gold, label_dict)
    # macro_f1 = scores['macro_f1']
    # micro_f1 = scores['micro_f1']
    # print('F 1******')
    # print('macro', macro_f1, 'micro', micro_f1)

# train_dataset = Dataset.from_dict({"inputs":batch[:80],"labels":labels[:80]})
# test_dataset = Dataset.from_dict({"inputs":batch[80:90],"labels":labels[80:90]})
# dev_dataset = Dataset.from_dict({"inputs":batch[90:],"labels":labels[90:]})

# train = DataLoader(train_dataset, batch_size=16, shuffle=False)
# dev = DataLoader(dev_dataset, batch_size=8, shuffle=False)


# def _precision_recall_f1(right, predict, total):
#     """
#     :param right: int, the count of right prediction
#     :param predict: int, the count of prediction
#     :param total: int, the count of labels
#     :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
#     """
#     p, r, f = 0.0, 0.0, 0.0
#     if predict > 0:
#         p = float(right) / predict
#     if total > 0:
#         r = float(right) / total
#     if p + r > 0:
#         f = p * r * 2 / (p + r)
#     return p, r, f


# def evaluate(epoch_predicts, epoch_labels, id2label, threshold=0.5, top_k=None):
#     """
#     :param epoch_labels: List[List[int]], ground truth, label id
#     :param epoch_predicts: List[List[int]], predicted label_id
#     :param vocab: data_modules.Vocab object
#     :param threshold: Float, filter probability for tagging
#     :param top_k: int, truncate the prediction
#     :return:  confusion_matrix -> List[List[int]],
#     Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
#     """
#     assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
#     epoch_gold = epoch_labels

#     # initialize confusion matrix
#     confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
#     right_count_list = [0 for _ in range(len(id2label))]
#     gold_count_list = [0 for _ in range(len(id2label))]
#     predicted_count_list = [0 for _ in range(len(id2label))]

#     for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
#         for i in range(len(confusion_count_list)):
#             for predict_id in sample_predict_id_list:
#                 confusion_count_list[i][predict_id] += 1

#         # count for the gold and right items
#         for gold in sample_gold:
#             gold_count_list[gold] += 1
#             for label in sample_predict_id_list:
#                 if gold == label:
#                     right_count_list[gold] += 1

#         # count for the predicted items
#         for label in sample_predict_id_list:
#             predicted_count_list[label] += 1

#     precision_dict = dict()
#     recall_dict = dict()
#     fscore_dict = dict()
#     right_total, predict_total, gold_total = 0, 0, 0

#     for i, label in id2label.items():
#         precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
#                                                                                             predicted_count_list[i],
#                                                                                             gold_count_list[i])
#         right_total += right_count_list[i]
#         gold_total += gold_count_list[i]
#         predict_total += predicted_count_list[i]

#     # Macro-F1
#     precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
#     recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
#     macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
#     # Micro-F1
#     precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
#     recall_micro = float(right_total) / gold_total
#     micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

#     return {'precision': precision_micro,
#             'recall': recall_micro,
#             'micro_f1': micro_f1,
#             'macro_f1': macro_f1,
#             'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}




# def print_params(module, input, output):
#     for name, param in module.named_parameters():
#         print(f'Parameter name: {name}, Parameter shape: {param.shape}, Parameter value: {param.data}')
# hook_handle = model.register_forward_hook(print_params)
