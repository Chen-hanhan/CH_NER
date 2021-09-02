import os
import re
import json
import logging
from transformers import BertTokenizer
from collections import defaultdict
import random

ENTITY_TYPES = ['器官组织', '阴性表现', '属性', '阳性表现', '否定描述', 
                '修饰描述', '异常现象', '数量', '测量值', '指代', '检查手段',
                '疾病','手术','期象', '累及部位', '病理分级', '病理分型', '病理分期']

class InputExample:
    def __init__(self, text,set_type, labels=None):
        self.text = text
        self.labels = labels # (start_index, end_index, type, entity)
        self.set_type = set_type


class BaseFeature:
    def __init__(
                self,
                input_ids,
                attention_masks,
                token_type_ids):
    # bert的输入
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class CRFFeature(BaseFeature):
    def __init__(
                self,
                input_ids,
                attention_masks,
                token_type_ids,
                labels
                ):
        super(CRFFeature, self).__init__(
            input_ids = input_ids,
            attention_masks = attention_masks,
            token_type_ids = token_type_ids 
        )
        self.labels = labels


class NERProcessor:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
    
    @staticmethod
    def read_data(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_text = []
            raw_labels = []

            for line in f:
                str_line = line.strip()
                p1 = re.compile(r'[{](.*?)[}]', re.S)
                str_line1 = re.findall(p1, str_line)
                if str_line1:
                    str_line1 = '{' + str_line1[0] +'}'
                    dict_line = eval(str_line1) # 字符串转为字典
                
                    for k, v in dict_line.items():
                        if k == 'sent':
                            v = v.replace(' ', '')
                            raw_text.append(v)
                        elif k == 'ners':
                            raw_labels.append(v)
                        else:
                            pass
        return raw_text, raw_labels
    
    def get_example(self, raw_text, raw_labels, set_type):
        examples = []
        for sent, labels in zip(raw_text, raw_labels):
            examples.append(InputExample(text = sent,
                                        labels = labels,
                                        set_type = set_type))

        return examples
    

def fine_grade_tokenize(raw_text, tokenizer: BertTokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []
    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)
    
    return tokens


def convert_crf_example(ex_idx, example: InputExample, max_seq_len,
                        tokenizer: BertTokenizer, ent2id):
    set_type = example.set_type
    raw_text = example.text
    entities = example.labels

    callback_info = (raw_text, )
    callback_labels = {x:[] for x in ENTITY_TYPES}

    for _label in entities:
        callback_labels[_label[2]].append((_label[3], _label[0]))

    callback_info += (callback_labels, )
    tokens = fine_grade_tokenize(raw_text, tokenizer)
    assert len(tokens) == len(raw_text)

    label_ids = None

    if set_type == 'train':
        # information for dev callback
        label_ids = [0] * len(tokens)

        # BIO标注
        for ent in entities:
            start_idx = ent[0]
            end_idx = ent[1]-1
            ent_type = ent[2]
            entity = ent[3]
            assert (start_idx + len(entity)-1) == end_idx
            # BIO
            if start_idx == end_idx:
                label_ids[start_idx] = ent2id['B-' + ent_type]
            else:
                label_ids[start_idx] = ent2id['B-' + ent_type]
                for i in range(start_idx+1, end_idx+1):
                    label_ids[i] = ent2id['I-' + ent_type]

            # # BIOES
            # if start_idx == end_idx:
            #     label_ids[start_idx] = ent2id['S' + ent_type]
            # else:
            #     label_ids[start_idx] = ent2id['B-' + ent_type]
            #     label_ids[end_idx] = ent2id['E-' + ent_type]
            #     for i in range(start_idx+1, end_idx):
            #         label_ids[i] = ent2id['I-' + ent_type]

        # cls
        if len(label_ids) > max_seq_len - 2:
            label_ids = label_ids[:max_seq_len - 2] 

        label_ids = [0] + label_ids + [0]

        # pad 因为要和tokenizer输出的input_ids对应
        if len(label_ids) < max_seq_len:
            pad_len = max_seq_len - len(label_ids)
            label_ids = label_ids + [0] * pad_len
        
        assert len(label_ids) == max_seq_len, f'{len(label_ids)}不等于{max_seq_len}'

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length = max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    feature = CRFFeature(
        input_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=label_ids
    )

    return feature, callback_info


def convert_examples_to_features(task_type, examples, max_seq_len, bert_dir, ent2id):
    assert task_type in ['crf']
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    features = []
    callback_info = []

    logging.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):
        if task_type == 'crf':
            feature, tmp_callback = convert_crf_example(
                ex_idx=i, example=example, tokenizer=tokenizer,
                max_seq_len=max_seq_len, ent2id=ent2id
            )

            features.append(feature)
            callback_info.append(tmp_callback)

    logging.info(f'Build {len(features)} features')

    out = (features, )

    if not len(callback_info):
        return out
    
    type_weight = {}  # 统计每一类的比例，用于计算 micro-f1
    for _type in ENTITY_TYPES:
        type_weight[_type] = 0.

    count = 0.

    if task_type == 'crf':
        for _callback in callback_info:
            for _type in _callback[1]:
                type_weight[_type] += len(_callback[1][_type])
                count += len(_callback[1][_type])

    for key in type_weight:
        type_weight[key] /= count

    out += ((callback_info, type_weight), )
    
    return out

if __name__ == '__main__':
    pass

