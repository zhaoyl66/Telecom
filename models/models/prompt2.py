# from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
from .modeling_bert import BertModel
from .configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from transformers.modeling_outputs import (
    MaskedLMOutput
)
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import os
from .loss import multilabel_categorical_crossentropy
from .graph import GraphEncoder
from .attention import CrossAttention
import math
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.autograd import Variable


class GraphEmbedding(nn.Module):
    def __init__(self, config, embedding, new_embedding, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEmbedding, self).__init__()
        self.graph_type = graph_type            #'GAT'
        padding_idx = config.pad_token_id
        self.num_class = config.num_labels
        if self.graph_type != '':
            self.graph = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)
        self.padding_idx = padding_idx
        self.original_embedding = embedding     #input_embedding

        new_embedding = torch.cat(
            [torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
             new_embedding], dim=0)
                                                #label_emb
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        # print('input_embedding:',self.original_embedding.num_embeddings ,'\tnew_embedding.num_embeddings',new_embedding.num_embeddings - 1)
        self.depth = (self.new_embedding.num_embeddings - 2 - self.num_class)

    @property
    def weight(self):
        def foo():
            # label prompt MASK
            edge_features = self.new_embedding.weight[1:, :]
            if self.graph_type != '':
                # label prompt
                edge_features = edge_features[:-1, :]
                edge_features = self.graph(edge_features, self.original_embedding)
                edge_features = torch.cat(
                    [edge_features, self.new_embedding.weight[-1:, :]], dim=0)
            return torch.cat([self.original_embedding.weight, edge_features], dim=0)

        return foo

    @property
    def raw_weight(self):
        def foo():
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)

        return foo

    def forward(self, x):
        x = F.embedding(x, self.weight(), self.padding_idx)

        return x


class OutputEmbedding(nn.Module):
    def __init__(self, bias):
        super(OutputEmbedding, self).__init__()
        self.weight = None
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class Prompt(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config,tokenizer=None, max_depth = 1,num_labels=1, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None, max_seg_num=None, max_seq_len=None, device=None, **kwargs):
        super().__init__(config)
        print('device:',device)
        # self.bert = BertModel(config, add_pooling_layer=False)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.tokenizer = tokenizer
        self.cls = BertOnlyMLMHead(config)
        self.num_labels = num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer
        # self.init_weights()

        #modeling_TSBert_v3     #跟多轮对话保持一致
        # self.bertConfig = BertConfig.from_pretrained(r'/data/zyl2023/bert-base-uncased',local_files_only=True)
        # self.bert = BertModel(BertConfig.from_pretrained(r'/data/zyl2023/bert-base-uncased',local_files_only=True))
        
        self.bertConfig = BertConfig.from_pretrained(r'/data/zyl2023/bert-base-chinese',local_files_only=True)
        self.bert = BertModel(BertConfig.from_pretrained(r'/data/zyl2023/bert-base-chinese',local_files_only=True))
        # self.config=bertConfig
        self.dropout = nn.Dropout(self.bertConfig.hidden_dropout_prob)
        self.alpha = 0.5
        # self.device=device
        self.linear_word = nn.Linear(2 * max_seq_len, 1) ##max_Seq_len
        self.W_word = nn.Parameter(data=torch.Tensor(self.bertConfig.hidden_size, self.bertConfig.hidden_size, max_seg_num))
        self.v = nn.Parameter(data=torch.Tensor(max_seg_num, 1))
        self.transformer_ur = TransformerBlock(device=device,input_size=self.bertConfig.hidden_size)
        self.transformer_ru = TransformerBlock(device=device,input_size=self.bertConfig.hidden_size)
        self.AU1 = nn.Parameter(data=torch.Tensor(self.bertConfig.hidden_size, self.bertConfig.hidden_size))
        self.AU3 = nn.Parameter(data=torch.Tensor(self.bertConfig.hidden_size, self.bertConfig.hidden_size))
        self.key_trans = nn.Linear(in_features=2*self.bertConfig.hidden_size, out_features=2*self.bertConfig.hidden_size)
        self.utt_gru_acc = nn.GRU(input_size=2*self.bertConfig.hidden_size, hidden_size=2*self.bertConfig.hidden_size, batch_first=True)
        # out_features: num_labels
        self.affine_out = nn.Linear(in_features=4*self.bertConfig.hidden_size, out_features=num_labels*max_depth)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # self.loss_func=BCELoss()
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_embedding(self):
        depth = len(self.depth2label)
        label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt'))
        # print("value_dict:",label_dict)
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.get_input_embeddings()
        for i in range(len(label_dict)):
            label_emb.append(
                input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id],
                                           device=self.device, dtype=torch.long))
        # prompt
        prompt_embedding = nn.Embedding(depth + 1,
                                        input_embeds.weight.size(1), 0)

        self._init_weights(prompt_embedding)
        # label prompt mask
        label_emb = torch.cat(
            [torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix], dim=0)
        embedding = GraphEmbedding(self.config, input_embeds, label_emb, self.graph_type,
                                   path_list=self.path_list, layer=self.layer, data_path=self.data_path)
        self.set_input_embeddings(embedding)
        output_embeddings = OutputEmbedding(self.get_output_embeddings().bias)
        self.set_output_embeddings(output_embeddings)
        output_embeddings.weight = embedding.raw_weight
        self.vocab_size = output_embeddings.bias.size(0)
        output_embeddings.bias.data = nn.functional.pad(
            output_embeddings.bias.data,
            (
                0,
                embedding.size - output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )

    def get_layer_features(self, layer, prompt_feature=None):
        labels = torch.tensor(self.depth2label[layer], device=self.device) + 1
        label_features = self.get_input_embeddings().new_embedding(labels)
        label_features = self.transform(label_features)
        label_features = torch.dropout(F.relu(label_features), train=self.training, p=self.config.hidden_dropout_prob)
        return label_features

    def forward(
            self,
            inputs=None,
            seg_input_ids=None,
            seg_token_type_ids=None,
            seg_attention_mask=None,
            cls_sep_pos=None,
            true_len=None,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        print("inputs size",inputs.size())
        L_seg_key = self.affine_out(inputs)
        multiclass_logits = torch.softmax(L_seg_key,dim=1).squeeze(dim=-1) #batch_size, max_depth*num_label
        # print('logits size:', multiclass_logits.size())
        print("self.num_labels",self.num_labels)
        logits = multiclass_logits.view(-1, self.num_labels)
        # print('forward logits:',logits.size())

        masked_lm_loss = None

        if labels is not None:
            multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), logits)

        return multiclass_loss,multiclass_logits

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate(self, inputs,labels, **kwargs):
        # attention_mask = input_ids != self.config.pad_token_id
        # print('before output')
        loss,logits = self(inputs=inputs,labels=labels)               ##调用forward 没有labels所以未计算labels的损失函数
        
        # multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        # prediction_scores = outputs['logits']
        # prediction_scores = prediction_scores.masked_select(
        #     multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
        #                                                                                   prediction_scores.size(
        #                                                                                       -1))
        # prediction_scores = prediction_scores[:,
        #                     self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
        # prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        
        # print('num_label:',self.num_labels)
        logits = logits.view(-1, self.num_labels) + self.multiclass_bias
        # print('logits:',logits.size())
        prediction_scores = logits.view(-1,len(self.depth2label),logits.size(-1))
        # print('predict_scores size: ',prediction_scores.size())
        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])
            for i, score in enumerate(scores):
                for l in self.depth2label[i]:
                    if score[l] > 0:
                        predict_labels[-1].append(l)
        # print('predict_labels:',len(predict_labels))
        # print('prediction_scores:',prediction_scores.size())
        return predict_labels, prediction_scores
    
    def word_selector(self, key, context,segment_turnmask):
        '''
        :param key:  (bsz, max_u_words, d)
        :param context:  (bsz,max_utterances, max_u_words, d)
        :param segment turn mask: (bsz,max_utterances)
        :return: score:
        '''
        # print("key.size():",key.size())
        # print("context.size()",context.size())
        '''
        context size 20 10 350 768
        key size 20 350 768
        segment turn mask 20 10
        W_word size 768 768 10
        A size: 20 10 350 350
        self.v max seg len
        '''
        dk = torch.sqrt(torch.Tensor([self.config.hidden_size])).to(self.device)
        A = torch.tanh(torch.einsum("blrd,ddh,bud->blruh", context, self.W_word, key)/dk)
        A = torch.einsum("blruh,hp->blrup", A, self.v).squeeze(dim=-1)   # b x l x u x u

        a = torch.cat([A.max(dim=2)[0], A.max(dim=3)[0]], dim=-1) # b x l x 2u
        a=self.linear_word(a).squeeze(dim=-1)
        mask=(1.0 - segment_turnmask) * -10000.0
        # print('mask:',mask.size(),mask)
        mask = Variable(mask, requires_grad=False)
        s1 = torch.softmax(a+mask, dim=-1)  # b x l
        # print('s1:',s1.size(),s1)
        return s1

    def utterance_selector(self, key, context,segment_turnmask):
        '''
        :param key:  (bsz, max_u_words, d)
        :param context:  (bsz,max_utterances, max_u_words, d)
        :return: score:
        '''
        key = key.mean(dim=1)
        # print('ut_select key:', key.size())
        context = context.mean(dim=2)
        # print('ut_select context:', context.size())
        s2 = torch.einsum("bud,bd->bu", context, key)/(1e-6 + torch.norm(context, dim=-1)*torch.norm(key, dim=-1, keepdim=True) )
        mask = (1.0 - segment_turnmask) * -10000.0
        # print('ut_select mask:', mask.size(),mask)
        mask = Variable(mask, requires_grad=False)
        s2 = torch.softmax(s2+mask, dim=-1)
        return s2

    def my_context_selector(self,segment,response,segment_turnmask):
        '''
        :param seg_context: (batch_size, max_segments, max_u_words, embedding_dim)
        :return:
        '''

        seg_score1 = self.word_selector(response,segment,segment_turnmask)
        seg_score2 = self.utterance_selector(response,segment,segment_turnmask)
        seg_score = self.alpha * seg_score1 + (1 - self.alpha) * seg_score2

        match_score_seg = seg_score

        select_seg_context = segment * match_score_seg.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return select_seg_context

    #Dual Cross-attention Matching 
    def MatchingNet(self,select_seg_context,segment_mask_seg,response_seg,response_mask_seg):
        '''

        :param select_seg_context: batchsize(*max_segment_num),seq_len,dim
        :param segment_mask_seg: batchsize(*max_segment_num),seq_len
        :param response_seg: batchsize(*max_segment_num),seq_len,dim
        :param response_mask_seg: batchsize(*max_segment_num),dim*2
        :return:
        '''
        Hur = self.transformer_ur(select_seg_context, response_seg, response_seg, response_mask_seg)    
        Hru = self.transformer_ru(response_seg,select_seg_context,select_seg_context,segment_mask_seg)  
        result=torch.cat([torch.mean(Hur,dim=1), torch.mean(Hru,dim=1) ], dim=1)
        # print('Hur size:',Hur.size(),'\tHru_size: ',Hru.size(),'\tresult:',result.size())
        
        return result


#modeling_TSBert_v3
def masked_softmax(vector, mask):
    mask = Variable(mask, requires_grad=False)
    result = torch.nn.functional.softmax(vector * mask, dim=-1)
    # a=(vector*mask).view(-1)
    # b=vector.view(-1)
    # for i, j in zip(a, b):
    #     if i != j:
    #         print(i, j)

    result = result * mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result

#Attentive Module
class TransformerBlock(nn.Module):

    def __init__(self, device,input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.device = device
        # print('TransformerBlock device:',device)
        self.init_weights()

    def init_weights(self):
        # init.xavier_normal_(self.linear1.weight)
        # init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V,V_mask, episilon=1e-8):
        '''
        :param Q: (batch_size*, max_r_words, embedding_dim)
        :param K: (batch_size*, max_u_words, embedding_dim)
        :param V: (batch_size*, max_u_words, embedding_dim)
        :param V_mask: (batch_size*, max_u_words)
        :return: output: (batch_size*, max_r_words, embedding_dim)  same size as Q
        '''
        dk = torch.Tensor([max(1.0, Q.size(-1))]).to(self.device)
        # print('dk:',dk.device)
        # print('Q:',Q.device)
        # print('K:',K.device)
        # print('V:',V.device)
        # print('V_mask:',V_mask.device)

        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon) #(batch_size, max_r_words, max_u_words)

        Q_K_score=masked_softmax(Q_K,V_mask[:,None,:])
        # Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_r_words, max_u_words)

        V_att = Q_K_score.bmm(V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        return output

