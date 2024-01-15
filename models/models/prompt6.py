from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
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


class GraphEmbedding(nn.Module):
    def __init__(self, config, embedding, new_embedding, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEmbedding, self).__init__()
        self.graph_type = graph_type
        padding_idx = config.pad_token_id
        self.num_class = config.num_labels
        if self.graph_type != '':
            self.graph = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)
        self.padding_idx = padding_idx
        self.original_embedding = embedding                     
        new_embedding = torch.cat(
            [torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
             new_embedding], dim=0)
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        print('input_embedding:',self.original_embedding.num_embeddings ,'\tnew_embedding.num_embeddings',self.new_embedding.num_embeddings - 1)
        # input_embedding.num_embeddings: 30522
        # new_embedding.num_embeddings: 57
        self.depth = (self.new_embedding.num_embeddings - 2 - self.num_class)
        print('depth:',self.depth)

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

    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None, **kwargs):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.cls = BertOnlyMLMHead(config)
        self.num_labels = config.num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_embedding(self):
        depth = len(self.depth2label)
        label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt'))
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.get_input_embeddings()
        for i in range(len(label_dict)):
            label_emb.append(
                input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
        # print('label_emb:',len(label_emb))  #54个label
        
        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id],
                                           device=self.device, dtype=torch.long))
        
        # prompt
        prompt_embedding = nn.Embedding(depth + 1,
                                        input_embeds.weight.size(1), 0)         #depth 2 + 1 *input_embeds.weigt
        # print('input_embeds.weight',input_embeds.weight)

        self._init_weights(prompt_embedding)
        # label prompt mask
        label_emb = torch.cat(
            [torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix], dim=0)
        # print('label prompt size:', label_emb.size())                               #57 768
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
            labels_depth=None,
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
        # print('labels size:',labels.size())                                                         # 16: batch_size 108: max_depth*num_class
        # print('input_ids size:',input_ids.size())                                                   # 16: batch_size 512:max_seq_len 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict       #True
        # print('return_dict:',return_dict)
        # print("self.get_input_embeddings()",self.get_input_embeddings())
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)                        # 16 512 获得CLS标记位置[PRED]
        non_false_indices = torch.nonzero(multiclass_pos != False).squeeze()
        # print('multiclass_pos:',non_false_indices)    
        # print('multiclass_pos:',multiclass_pos.size())
        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100)
        # print('single_labels:',single_labels)                                                       #CLS & padding token变为-100
        
        #随机对token进行mask 随机替换 
        if self.training:
            enable_mask = input_ids < self.tokenizer.vocab_size                     #超过词汇表大小的词 False=0
            # print('enable_mask:',enable_mask.size())
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
            # print('random_mask:',random_mask.size())                                       #padding token [0]   超过词汇表的token [0]
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)    #[MASK]
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)        #input_ids 大小 [104, vocab_size]随机元素
            mlm_mask = random_mask > 0.985                                          #>0.985 True 1; <=0.985 False 0
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask  #>0.985 random_ids; <=0.985 input_ids
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

        # print('input size:',input_ids.size())               #size 16 512
        sequence_output = outputs[0]
        # print('sequence_output:',sequence_output.size())    #16 512+1(L_seg_key) 768


        prediction_scores = self.cls(sequence_output)       #CLS位置 分类scores 16 512 768
        # print('prediction :',prediction_scores.size())      #size 16 512 30579:vocab_id 看一下
        # print('prediction:', prediction_scores[0][0][self.vocab_size:])

        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)),
                                      single_labels.view(-1))

            multiclass_logits = prediction_scores.masked_select(
                multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,prediction_scores.size(-1))
            # print('size :',multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1)).size())      # 16 512 30579
            print('multiclass_logits:',multiclass_logits.size())   # 32 30579 batch size * max_depth  vocab_size
            # print("multiclass_pos.unsqueeze(-1)",multiclass_pos.unsqueeze(-1).size(),multiclass_pos.unsqueeze(-1))
            # print("multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))",multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1)).size(),multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1)))
            # print("prediction scores masked select", prediction_scores.masked_select(multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).size(),prediction_scores.masked_select(multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))))
            # print("vocab size",self.vocab_size)                                             
            multiclass_logits = multiclass_logits[:,
                                self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
            # print("multiclass_logits[0]",)
            print(labels.size())
            new_labels_list = []
            for idx,i in enumerate(labels_depth):
                label = labels[idx][:i*self.num_labels]
                label = label.view(-1, self.num_labels)
                new_labels_list.append(label)
            new_labels = torch.cat(new_labels_list,dim=0)
            print('new_labels:',new_labels.size())
            print('multiclass_logits:',multiclass_logits.size(),'\tlabels:',labels.view(-1, self.num_labels).size())  
            #32 54:label_num
            #多标签损失函数
            # multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), multiclass_logits)
            multiclass_loss = multilabel_categorical_crossentropy(new_labels, multiclass_logits)
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
        # print('ret size:',ret)
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
    def generate(self, input_ids, depth2label, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        # print('before output')
        outputs = self(input_ids, attention_mask)
        # print('after output')
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        prediction_scores = outputs['logits']
        prediction_scores = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                          prediction_scores.size(
                                                                                              -1))
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
        return predict_labels, prediction_scores
