# Adopt from https://github.com/Alibaba-NLP/HiAGM/blob/master/train_modules/evaluation_metrics.py


def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f

# def golden_transfer_within_tolerance_exp(
#     pre_labels, true_labels, t=1, eps=1e-7, lamb=0
# ):
#     print("pre_labels len:",len(pre_labels))
#     print("true_labels len:",len(true_labels))
#     print("pre_labels ",pre_labels)
#     print("true_labels ",true_labels)
#     if t <= 0:
#         raise ValueError("Tolerance must be positive!!!")
#     if not isinstance(t, int):
#         raise TypeError("Tolerance must be Integer!!!")

#     gtt_score = 0
#     suggest_indices = []
#     for idx, label in enumerate(true_labels):
#         if label == 1:
#             print("suggest_indices",idx)
#             suggest_indices.append(idx)

#     pre_indices = []
#     for idx, label in enumerate(pre_labels):
#         if label == 1:
#             print("pre_indices",idx)
#             pre_indices.append(idx)

#     if len(suggest_indices) == 0:
#         if len(pre_indices) == 0:
#             gtt_score = 1
#         else:
#             gtt_score = 0
#     else:
#         if len(pre_indices) == 0:
#             gtt_score = 0
#         else:
#             GST_score_list = []
#             for pre_idx in pre_indices:
#                 tmp_score_list = []
#                 for suggest_idx in suggest_indices:
#                     # suggest_idx is q_i
#                     # pre_idx is p_i
#                     pre_bias = pre_idx - suggest_idx
#                     adjustment_cofficient = 1.0 / (1 - lamb * (np.sign(pre_bias)))
#                     tmp_score = math.exp(
#                         -(adjustment_cofficient)
#                         * math.pow(pre_bias, 2)
#                         / (2 * math.pow((t + eps), 2))
#                     )
#                     tmp_score_list.append(tmp_score)
#                 GST_score_list.append(np.max(tmp_score_list))
#             # print(punishment_ratio)
#             gtt_score = np.mean(GST_score_list)
#     return gtt_score

# def compute_GT(pre_labels, true_labels, tolerance=1):
#     num_samples = len(pre_labels)
#     gt_scores = []

#     for i in range(num_samples):
#         pre = pre_labels[i]
#         true = true_labels[i]
#         max_score = 0.0

#         for true_label in true:
#             max_tmp = 0.0

#             for pre_label in pre:
#                 pre_bias = pre_label - true_label
#                 tmp_score = math.exp(-0.5 * (pre_bias/tolerance)**2)
#                 max_tmp = max(max_tmp, tmp_score)

#             max_score += max_tmp

#         gt_scores.append(max_score / len(true))

#     gt_score = sum(gt_scores) / num_samples
#     return gt_score

def evaluate(epoch_predicts, epoch_labels, id2label, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[int]], predicted label_id
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    # label2id = vocab.v2i['label']
    # id2label = vocab.i2v['label']
    # epoch_gold_label = list()
    # # get id label name of ground truth
    # for sample_labels in epoch_labels:
    #     sample_gold = []
    #     for label in sample_labels:
    #         assert label in id2label.keys(), print(label)
    #         sample_gold.append(id2label[label])
    #     epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        # Here we observe some classes will not appear in the test set and scores of these classes are set to 0.
        # If we exclude those classes, Macro-F1 will dramatically increase.
        # if gold_count_list[i] + predicted_count_list[i] != 0:
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        # print('label:',label,'\tfscore_dict[label]:',fscore_dict[label])
        # fscore 每个类别的F1值和hpt单独的比较一下
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0
    # gt_1 = golden_transfer_within_tolerance_exp(epoch_predicts,epoch_labels)

    return {
            'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            # 'gt_1' : gt_1,
            'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}
