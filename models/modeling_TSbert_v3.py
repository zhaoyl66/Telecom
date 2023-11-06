import torch
from torch import nn
from torch.nn import BCELoss
from .modeling_bert import BertPreTrainedModel,BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.functional as F
from torch.autograd import Variable


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


class BertForSequenceClassificationTSv3(BertPreTrainedModel):
    def __init__(self, config, max_seg_num, max_seq_len, device):
        super(BertForSequenceClassificationTSv3, self).__init__(config)
        self.config=config
        # self.seq_len=seq_len
        # self.finaldim=finaldim
        # self.num_labels = num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.alpha = 0.5
        self.device=device
        self.linear_word = nn.Linear(2 * max_seq_len, 1) ##max_Seq_len
        self.W_word = nn.Parameter(data=torch.Tensor(self.config.hidden_size, self.config.hidden_size, max_seg_num))
        self.v = nn.Parameter(data=torch.Tensor(max_seg_num, 1))

        self.transformer_ur = TransformerBlock(device=device,input_size=self.config.hidden_size)
        self.transformer_ru = TransformerBlock(device=device,input_size=self.config.hidden_size)
        self.AU1 = nn.Parameter(data=torch.Tensor(self.config.hidden_size, self.config.hidden_size))
        # self.AU2 = nn.Parameter(data=torch.Tensor(self.config.hidden_size, self.config.hidden_size))
        self.AU3 = nn.Parameter(data=torch.Tensor(self.config.hidden_size, self.config.hidden_size))

        # self.utt_cnn_2d_1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3, 3))
        # self.utt_maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.utt_cnn_2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        # self.utt_maxpooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.utt_cnn_2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        # self.utt_maxpooling3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.utt_affine2 = nn.Linear(in_features=3 * 3 * 64, out_features=self.finaldim)
        self.key_trans = nn.Linear(in_features=2*self.config.hidden_size, out_features=2*self.config.hidden_size)

        self.utt_gru_acc = nn.GRU(input_size=2*self.config.hidden_size, hidden_size=2*self.config.hidden_size, batch_first=True)
        self.affine_out = nn.Linear(in_features=4*self.config.hidden_size, out_features=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.loss_func=BCELoss()
        self.init_weights()

    # def my_init_weights(self):
    #     init.uniform_(self.W_word)
    #     init.uniform_(self.v)
    #     init.uniform_(self.linear_word.weight)
    #     # init.uniform_(self.linear_score.weight)
    #
    #     init.xavier_normal_(self.AU1)
    #     init.xavier_normal_(self.AU2)
    #     init.xavier_normal_(self.AU3)
    #     init.xavier_normal_(self.utt_cnn_2d_1.weight)
    #     init.xavier_normal_(self.utt_cnn_2d_2.weight)
    #     init.xavier_normal_(self.utt_cnn_2d_3.weight)
    #     init.xavier_normal_(self.utt_affine2.weight)
    #     for weights in [self.utt_gru_acc.weight_hh_l0, self.utt_gru_acc.weight_ih_l0]:
    #         init.orthogonal_(weights)
    #     init.xavier_normal_(self.key_trans.weight)
    #
    #     init.xavier_normal_(self.affine_out.weight)

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
        print('mask:',mask.size(),mask)
        mask = Variable(mask, requires_grad=False)
        s1 = torch.softmax(a+mask, dim=-1)  # b x l
        print('s1:',s1.size(),s1)
        return s1

    def utterance_selector(self, key, context,segment_turnmask):
        '''
        :param key:  (bsz, max_u_words, d)
        :param context:  (bsz,max_utterances, max_u_words, d)
        :return: score:
        '''
        key = key.mean(dim=1)
        print('ut_select key:', key.size())
        context = context.mean(dim=2)
        print('ut_select context:', context.size())
        s2 = torch.einsum("bud,bd->bu", context, key)/(1e-6 + torch.norm(context, dim=-1)*torch.norm(key, dim=-1, keepdim=True) )
        mask = (1.0 - segment_turnmask) * -10000.0
        print('ut_select mask:', mask.size(),mask)
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
        print('Hur size:',Hur.size(),'\tHru_size: ',Hru.size(),'\tresult:',result.size())
        
        return result

    def forward(self, seg_input_ids,seg_token_type_ids, seg_attention_mask,cls_sep_pos,true_len,labels=None):
        self.utt_gru_acc.flatten_parameters()
        b, s = seg_input_ids.size()
        # b: batch_size 20
        # s: max_seq 350
        n = cls_sep_pos.size()[1] - 2
        print('cls_sep: ',cls_sep_pos.size()) #全部是[20,12] 
        print('n:',n)                         #全部是 10 最大topic segment数


        sequence_output, pooled_output = self.bert(input_ids=seg_input_ids,
                                                   attention_mask=seg_attention_mask,
                                                   token_type_ids=seg_token_type_ids)
        print('input_ids_size:',seg_input_ids.size())           #[20,350] batch_size, max_seq
        print('cls_sep_pos_size:',cls_sep_pos.size())           #[20,12]  batch_size, seg_num
        print('sequence_output_size: ', sequence_output.size()) #[20,350,768] batch_size max_seq dim

        _,_,d=sequence_output.size()
        # b,max_seg_num,max_seq_len,dim
        segment = torch.zeros(b,n,s, d).to(self.device)         #20 10 350 768
        segment_mask = torch.zeros(b , n, s).to(self.device)    #20 10 350
        segment_turnmask=torch.zeros(b , n).to(self.device)     #20 10

        # b,max_seq_len,dim
        response = torch.zeros(b, s, d).to(self.device)         #20 350 768
        response_mask = torch.zeros(b , s).to(self.device)      #20 350

        for bind,seq in enumerate(sequence_output):
            print('true_len: ',true_len[bind])
            cls_seq_pos_temp=cls_sep_pos[bind][:true_len[bind]] 
            print('cls_seq_pos_temp',cls_seq_pos_temp)          #[0, 36, 48] cls_sep_pos去掉-1
            for posind,pos in enumerate(cls_seq_pos_temp):
                if(posind==true_len[bind]-1):
                    break
                m = cls_seq_pos_temp[posind + 1] - cls_seq_pos_temp[posind] - 1   #[cls][sep]间
                if(posind==true_len[bind]-2):                   #response
                    response[bind][0:m] = sequence_output[bind][cls_seq_pos_temp[posind] + 1:cls_seq_pos_temp[posind + 1]]
                    response_mask[bind][0:m]= 1
                    print('response[bind][0:m]',response[bind][0:m].size())
                    print('response_mask[bind]',response_mask[bind][0:m].size())
                else:                                           #context segment
                    segment[bind][posind][0:m]=\
                        sequence_output[bind][cls_seq_pos_temp[posind]+1:cls_seq_pos_temp[posind+1]]
                    segment_mask[bind][posind][0:m] =1
                    segment_turnmask[bind][posind]=1
                


        # utt_sequence_output=utt_sequence_output.view(b,u,w,d) #[batch_size,max_utts,max_words,dim]
        # seg_sequence_output=seg_sequence_output.view(b,s,w,d) #[batch_size,max_utt_len,max_words,dim]
        # #res_sequence_output [batch_size,max_words,dim]

        #segment b,n,s,d mask n,n,s
        #response b.s.d mask b,s
        # m = torch.gt(segment * segment_mask.unsqueeze(dim=-1), segment)
        # w=(segment * segment_mask.unsqueeze(dim=-1)).view(-1)
        # e=segment.view(-1)
        # for i,j in zip(a,b):
        #     if i!=j:
        #         print(i,j)
        
        
        # segment size 20 10 350 768
        # response size 20 350 768
        # segment turn mask 20 10
        # score 20 10
        
        #Segment Weighting
        select_seg_context = self.my_context_selector(segment,response,segment_turnmask) # 20 10 350 768
        print('select_seg_context: ',select_seg_context.size())

        ''' m = torch.gt(select_seg_context * segment_mask.unsqueeze(dim=-1), select_seg_context)
        # w = (select_seg_context * segment_mask.unsqueeze(dim=-1)).view(-1)
        # e = select_seg_context.view(-1)
        # for i, j in zip(a, b):
        #     if i != j:
        #         print(i, j)
        # select_utt_context=select_utt_context.view(-1,w,d)'''

        select_seg_context=select_seg_context.view(b*n,s,d)                     #200 350 768
        segment_mask_seg = segment_mask.view(b * n, s)                          #200 350


        # res_sequence_output_utt=res_sequence_output.unsqueeze(dim=1).repeat(1, u, 1, 1)
        # res_sequence_output_utt=res_sequence_output_utt.view(-1,w,d)
        res_sequence_output_seg= response.unsqueeze(dim=1).repeat(1, n, 1, 1)
        response_mask_seg=response_mask.unsqueeze(dim=1).repeat(1, n, 1, 1)    #1 200 1 350
        print('response_mask_seg: ',response_mask_seg.size())
        res_sequence_output_seg=res_sequence_output_seg.view(b*n,s,d)           #200 350 768
        response_mask_seg = response_mask_seg.view(b * n, s)                    #200 350
        print('response_mask_seg after: ',response_mask_seg.size())

        # V_utt = self.UR_Matching_utt(select_utt_context, res_sequence_output_utt)
        # V_utt = V_utt.view(b, u, -1)  # (bsz, max_utterances, 300)

        V_seg = self.MatchingNet(select_seg_context,segment_mask_seg,res_sequence_output_seg,response_mask_seg)
        print('V_seg size:', V_seg.size())  #200 1536
        V_seg=V_seg.view(b, n, 2*d)         #20 10 1536
        #V_seg size 20 10 1536 : 768*2

        #segment 20 10 350 768  取最后一个topic segment 沿着topic segment维度进行平均
        V_key = self.MatchingNet(segment[:, -1:, :, :].mean(dim=1), segment_mask[:, -1:, :].mean(dim=1), response,response_mask)  # (bsz,2dim)
        print('V_key size:', V_key.size())  # 20 1536


        ''' H_utt, _ = self.utt_gru_acc(V_utt)  # (bsz, max_utterances, rnn2_hidden)'''
        H_seg, _ = self.utt_gru_acc(V_seg)  # (bsz, max_segments, rnn2_hidden)
        print('H_seg:',H_seg.size(),H_seg)  # 20 10 1536
        
        #key_trans 自定义线性层 input 20 1536 output 20 1536
        H_key = self.key_trans(V_key)
        print('H_key: ',H_key.size(),H_key) #20 1536

        ''' L = self.attention(V, u_mask_sent)
        # L_utt = self.dropout(H_utt[:, -1, :])  # (bsz, rnn2_hidden)'''
        L_seg = self.dropout(H_seg[:, -1, :])  # (bsz, rnn2_hidden)
        L_key = self.dropout(H_key)  # (bsz, rnn2_hidden)
        print('L_seg:',L_seg.size(),'\tL_key:',L_key.size())    # 20 1536
        print('L_seg+L_key:',torch.cat((L_seg, L_key), 1).size())   #20 3072


        logits = torch.sigmoid(self.affine_out(torch.cat((L_seg, L_key), 1))).squeeze(dim=-1)
        # labels=torch.FloatTensor(labels)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss = self.loss_func(logits, target=labels)
            return logits,loss
        else:
            return logits

