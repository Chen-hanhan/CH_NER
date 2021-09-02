import os
import math
import torch
import torch.nn as nn
from torchcrf import CRF
# import torchcrf
from itertools import repeat
from transformers import BertModel
from src.utils.evaluator import crf_decode
# from src.utils.functions_utils import vote
# from src.utils.evaluator import crf_decode, span_decode



class BaseModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob) -> None:
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')
        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
           'pretrained bert file does not exist'
        
        self.bert_module = BertModel.from_pretrained(bert_dir,
                                                    hidden_dropout_prob=dropout_prob,
                                                    output_hidden_states=True)

        self.bert_config = self.bert_module.config
        
    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))# 正态分布
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


class CRFModel(BaseModel):
    def __init__(self, bert_dir, num_tags,
                dropout_prob=0.1, **kwargs) -> None:
        super().__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)
        
        bert_out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(bert_out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        out_dims = mid_linear_dims

        self.classifier = nn.Linear(out_dims, num_tags)

        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.crf_module = CRF(num_tags=num_tags, batch_first=True)

        # 初始化
        init_blocks = [self.mid_linear, self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)


    def forward(self, token_ids, attention_masks, token_type_ids,
                labels=None):
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        seq_out = bert_outputs[0]

        mid_linear_out = self.mid_linear(seq_out)

        emissions = self.classifier(mid_linear_out)

        if labels is not None:
            tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                tags=labels.long(),
                                                mask=attention_masks.byte(),
                                                reduction='mean')

            out = (tokens_loss, )
        else:
            tokens_out = self.crf_module.decode(emissions=emissions,
                                                mask=attention_masks.byte())

            out = (tokens_out, emissions)
        
        return out


def build_model(task_type, bert_dir, **kwargs):
    assert task_type in ['crf']

    if task_type == 'crf':
        model = CRFModel(bert_dir=bert_dir,
                        num_tags=kwargs.pop('num_tags'),
                        dropout_prob=kwargs.pop('dropout_prob', 0.1))

    return model
