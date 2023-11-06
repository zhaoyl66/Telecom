import logging
import numpy as np
logger = logging.getLogger(__name__)
import json


def convert_cs_to_feature(prefix,
                        context,
                        response,
                        max_seg_num,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=False,
                        cls_token='[CLS]',
                        cls_token_segment_id=1,
                        sep_token='[SEP]',
                        sep_token_extra=False,
                        pad_on_left=False,
                        pad_token=0,
                        pad_token_segment_id=0,
                        sequence_a_segment_id=0,
                        sequence_b_segment_id=0,
                        prefix_segment_id=1,
                        mask_padding_with_zero=True):
    
    tokens_a = []
    for seg in context[-max_seg_num:]:
        tokens_a += tokenizer.tokenize(seg) + [sep_token]
    tokens_a = tokens_a[:-1]  # 去掉最后一个[SEP]       s1[SEP]s2[SEP]s3[SEP]S4
    # print(len(tokens_a))
    if len(tokens_a) == 0:
        print('context',context)
        print('response',response)
        print('tokens_a:',tokens_a)
    # print('context',context)
    # print('response',response)
    tokens_b = tokenizer.tokenize(response)
    special_tokens_count = 4 if sep_token_extra else 3
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

    if (tokens_a[0] == sep_token):
        tokens_a = tokens_a[1:]
    tokens = tokens_a + [sep_token]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
    
    segment_ids += [prefix_segment_id] * (len(prefix))

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids     #[CLS]s1[SEP]s2[SEP]s3[SEP]S4[SEP]res[SEP]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)    
    
    input_mask = [1 if mask_padding_with_zero else 0] * (len(input_ids) + len(prefix))

    cls_sep_pos = [0]

    for tok_ind, tok in enumerate(tokens):
        if (tok == sep_token):
            cls_sep_pos.append(tok_ind)
    true_len = len(cls_sep_pos)
    while (len(cls_sep_pos) < max_seg_num + 2):
        cls_sep_pos.append(-1)
    
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + prefix + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
    assert len(input_ids) == max_seq_length + len(prefix)
    assert len(input_mask) == max_seq_length + len(prefix)
    assert len(segment_ids) == max_seq_length + len(prefix)
    feature = {'seg_input_ids':input_ids ,'seg_token_type_ids':segment_ids,'seg_attention_mask':input_mask,'cls_sep_pos':cls_sep_pos,'true_len':true_len}

    return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()