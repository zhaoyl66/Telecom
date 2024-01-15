# from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
# from .modeling_bert import BertModel
from .configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead, BertModel
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
import sys
import torch.nn.init as init


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
    def __init__(self, config,bias):
        super(OutputEmbedding, self).__init__()
        self.weight = None
        self.bias = bias
        self.Linear = nn.Linear(config.hidden_size,config.hidden_size)

    def forward(self, x):
        return F.linear(self.Linear(x), self.weight(), self.bias)


class Prompt(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config,tokenizer=None, max_depth = 1,num_labels=1, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None, max_seg_num=None, max_seq_len=None, device=None, **kwargs):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.cls = BertOnlyMLMHead(config)
        self.num_labels = num_labels
        self.max_depth = max_depth
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        print("self.tokenizer.vocab_size",self.tokenizer.vocab_size)
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer
        self.init_weights()

        self.bertConfig = BertConfig.from_pretrained(r'/data/zyl2023/bert-base-chinese',local_files_only=True)
        # self.bert = BertModel(BertConfig.from_pretrained(r'/data/zyl2023/bert-base-chinese',local_files_only=True))
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.bertConfig.hidden_dropout_prob)
        self.alpha = 0.5
        self.linear_word = nn.Linear(2 * max_seq_len, 1) ##max_Seq_len
        self.W_word = nn.Parameter(data=torch.Tensor(self.bertConfig.hidden_size, self.bertConfig.hidden_size, max_seg_num))
        self.v = nn.Parameter(data=torch.Tensor(max_seg_num, 1))
        self.transformer_ur = TransformerBlock(device=device,input_size=self.bertConfig.hidden_size)
        self.transformer_ru = TransformerBlock(device=device,input_size=self.bertConfig.hidden_size)
        self.key_trans = nn.Linear(in_features=2*self.bertConfig.hidden_size, out_features=2*self.bertConfig.hidden_size)
        self.utt_gru_acc = nn.GRU(input_size=2*self.bertConfig.hidden_size, hidden_size=2*self.bertConfig.hidden_size, batch_first=True)
        self.affine_out = nn.Linear(in_features=4*self.bertConfig.hidden_size, out_features=self.bertConfig.hidden_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.init_weights()
        self.labels = []

    def get_output_embeddings(self):
        print("self.cls.predictions",self.cls.predictions.decoder)
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_embedding(self):
        depth = len(self.depth2label)
        label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt'))
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.get_input_embeddings()      # bertModel的input_embedding
        for i in range(len(label_dict)):
            label_emb.append(
                input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
        # print("label_emb",len(label_emb)) #长度 141
        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id],
                                           device=self.device, dtype=torch.long))
        # print("prefix",prefix.size())   #1 768
        # prompt
        prompt_embedding = nn.Embedding(depth + 1,
                                        input_embeds.weight.size(1), 0)
        # print("prompt_embedding",prompt_embedding)  # 3=max_depth + 1 * 768 padding_ids=0

        self._init_weights(prompt_embedding)
        label_emb = torch.cat(
            [torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix], dim=0)
        # 141(label_num) + 3(prompt_embedding max_depth + 1) + 1(prefix)
        
        embedding = GraphEmbedding(self.config, input_embeds, label_emb, self.graph_type,
                                   path_list=self.path_list, layer=self.layer, data_path=self.data_path)
        
        self.set_input_embeddings(embedding)
        output_embeddings = OutputEmbedding(self.config,self.get_output_embeddings().bias)
        self.set_output_embeddings(output_embeddings)                        
        output_embeddings.weight = embedding.raw_weight  #raw_weight()方法 21272
        self.vocab_size = output_embeddings.bias.size(0)
        print("self.vocab_size ",self.vocab_size)
        #对模型偏置项进行填充0
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
            seg_input_ids=None,
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
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict       #True
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)                      #获得CLS标记位置
        # print('multiclass_pos:',multiclass_pos.size())                                                  #20 512
        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100) #[mask]
        # print('single_labels:',single_labels.size())                                                      #CLS & padding token变为-100
        
        #随机对token进行mask 随机替换 
        if self.training:
            enable_mask = input_ids < self.tokenizer.vocab_size                     #超过词汇表大小的词 False=0
            # print('enable_mask:',enable_mask[0])
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * seg_attention_mask * enable_mask
            # print('random_mask:',random_mask)                                       #padding token [0]   超过词汇表的token [0]
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)    #[MASK]
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)        #input_ids 大小 [104, vocab_size]随机元素
            mlm_mask = random_mask > 0.985                                          #>0.985 True 1; <=0.985 False 0
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask  #>0.985 random_ids; <=0.985 input_ids
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)

        num = single_labels.shape[0]
        multiturn_labels = torch.full((num,1),-100).to(self.device)
        single_labels = torch.cat((single_labels,multiturn_labels),dim=1)
        multirturn_pos =  torch.full((num,1),False).to(self.device)
        multiclass_pos = torch.cat((multirturn_pos,multiclass_pos),dim=1)

        # print('input_ids',input_ids)
        # print('attention_mask',attention_mask)
        # print('token_type_ids',token_type_ids)
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask= attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        self.utt_gru_acc.flatten_parameters()
        b, s = seg_input_ids.size()
        # b: batch_size 16
        # s: max_seq 350
        n = cls_sep_pos.size()[1] - 2

        _,_,d=sequence_output.size()
        # batch_size,max_seg_num,max_seq_len,dim
        segment = torch.zeros(b,n,s, d).to(self.device)
        segment_mask = torch.zeros(b , n, s).to(self.device)
        segment_turnmask=torch.zeros(b , n).to(self.device) 

        # b,max_seq_len,dim
        response = torch.zeros(b, s, d).to(self.device)
        response_mask = torch.zeros(b , s).to(self.device)

        for bind,seq in enumerate(sequence_output):
            cls_seq_pos_temp=cls_sep_pos[bind][:true_len[bind]] 
            for posind,pos in enumerate(cls_seq_pos_temp):
                if(posind==true_len[bind]-1):
                    break
                m = cls_seq_pos_temp[posind + 1] - cls_seq_pos_temp[posind] - 1 
                if(posind==true_len[bind]-2):                   #response
                    response[bind][0:m] = sequence_output[bind][cls_seq_pos_temp[posind] + 1:cls_seq_pos_temp[posind + 1]]
                    response_mask[bind][0:m]= 1
                else:
                    segment[bind][posind][0:m]=\
                        sequence_output[bind][cls_seq_pos_temp[posind]+1:cls_seq_pos_temp[posind+1]]
                    segment_mask[bind][posind][0:m] =1
                    segment_turnmask[bind][posind]=1
        
        #Segment Weighting
        select_seg_context = self.my_context_selector(segment,response,segment_turnmask)

        select_seg_context=select_seg_context.view(b*n,s,d)
        segment_mask_seg = segment_mask.view(b * n, s)

        res_sequence_output_seg= response.unsqueeze(dim=1).repeat(1, n, 1, 1)
        response_mask_seg=response_mask.unsqueeze(dim=1).repeat(1, n, 1, 1)

        res_sequence_output_seg=res_sequence_output_seg.view(b*n,s,d)
        response_mask_seg = response_mask_seg.view(b * n, s)
        
        V_seg = self.MatchingNet(select_seg_context,segment_mask_seg,res_sequence_output_seg,response_mask_seg)

        V_seg=V_seg.view(b, n, 2*d)

        V_key = self.MatchingNet(segment[:, -1:, :, :].mean(dim=1), segment_mask[:, -1:, :].mean(dim=1), response,response_mask)  # (bsz,2dim)

        H_seg, _ = self.utt_gru_acc(V_seg)
        H_key = self.key_trans(V_key)

        L_seg = self.dropout(H_seg[:, -1, :])       # (bsz, rnn2_hidden)

        L_key = self.dropout(H_key)                 # (bsz, rnn2_hidden) 
        L_seg_key = self.affine_out(torch.cat((L_seg, L_key), 1))
        L = torch.unsqueeze(L_seg_key,dim=1) 
        sequence_output = torch.cat((sequence_output,L),1)
        prediction_scores = self.cls(sequence_output)
        
        masked_lm_loss = None

        if labels is not None:
            
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)),
                                      single_labels.view(-1))
            #select PRED
            multiclass_logits = prediction_scores.masked_select(
                multiclass_pos.unsqueeze(-1).expand(-1,-1, prediction_scores.size(-1))).view(-1,prediction_scores.size(-1))
            multiclass_logits = multiclass_logits[:,
                                self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias      #num_labels
            
            multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), multiclass_logits)

            masked_lm_loss += multiclass_loss  

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        ret = MaskedLMOutput(
            loss=masked_lm_loss,                
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return ret

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
    def generate(self, input_ids,attention_mask,token_type_ids, seg_token_type_ids, seg_attention_mask, cls_sep_pos, true_len, depth2label, **kwargs):
        # print('generate token_type_ids',token_type_ids)
        outputs = self(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids, seg_token_type_ids=seg_token_type_ids, seg_attention_mask=seg_attention_mask, cls_sep_pos=cls_sep_pos,true_len=true_len)               ##调用forward 没有labels所以未计算labels的损失函数
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        # print('multiclass_pos',multiclass_pos.unsqueeze(-1))  
        num = multiclass_pos.shape[0]
        multirturn_pos =  torch.full((num,1),False).to(self.device)
        multiclass_pos = torch.cat((multirturn_pos,multiclass_pos),dim=1)
        prediction_scores = outputs['logits']
        prediction_scores = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,prediction_scores.size(-1))
        prediction_scores = prediction_scores[:,
                            self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
        prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])
            for i, score in enumerate(scores):
                for l in depth2label[i]:
                    if score[l] > 0:
                        predict_labels[-1].append(l)
            self.labels.append(predict_labels[-1])

        return predict_labels, prediction_scores
    
    def word_selector(self, key, context,segment_turnmask):
        '''
        :param key:  (bsz, max_u_words, d)
        :param context:  (bsz,max_utterances, max_u_words, d)
        :param segment turn mask: (bsz,max_utterances)
        :return: score:
        '''
        dk = torch.sqrt(torch.Tensor([self.config.hidden_size])).to(self.device)
        A = torch.tanh(torch.einsum("blrd,ddh,bud->blruh", context, self.W_word, key)/dk)
        A = torch.einsum("blruh,hp->blrup", A, self.v).squeeze(dim=-1)   # b x l x u x u

        a = torch.cat([A.max(dim=2)[0], A.max(dim=3)[0]], dim=-1) # b x l x 2u
        a=self.linear_word(a).squeeze(dim=-1)
        mask=(1.0 - segment_turnmask) * -10000.0
        mask = Variable(mask, requires_grad=False)
        s1 = torch.softmax(a+mask, dim=-1)
        return s1

    def utterance_selector(self, key, context,segment_turnmask):
        '''
        :param key:  (bsz, max_u_words, d)
        :param context:  (bsz,max_utterances, max_u_words, d)
        :return: score:
        '''
        key = key.mean(dim=1)
        context = context.mean(dim=2)
        s2 = torch.einsum("bud,bd->bu", context, key)/(1e-6 + torch.norm(context, dim=-1)*torch.norm(key, dim=-1, keepdim=True) )
        mask = (1.0 - segment_turnmask) * -10000.0
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
        
        return result


#modeling_TSBert_v3
def masked_softmax(vector, mask):
    mask = Variable(mask, requires_grad=False)
    result = torch.nn.functional.softmax(vector * mask, dim=-1)

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
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

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

