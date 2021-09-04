import re
import numpy as np
import random

ENTITY_TYPES = ['指代', '手术','期象', '累及部位', '病理分级', '病理分型', '病理分期']
ENTITY_TYPES_PY = ['zhidai', 'shoushu', 'qixiang', 'leijibuwei', 'binglifen_ji', 'binglifen_xing', 'binglifen_qi']
ENTITY = ['ent_zhidai', 'ent_shoushu', 'ent_qixiang', 'ent_leijibuwei', 'ent_binglifen_ji', 'ent_binglifen_xing', 'ent_binglifen_qi']



def get_example_sent_ner(file_dir): # return sent, ner_label
    with open(file_dir, encoding='utf-8') as conll:
        sent = [] # 完整句子
        ner_label = [] # 标签
        for line in conll:
            str_line = line.strip()
            p1 = re.compile(r'[{](.*?)[}]', re.S)
            str_line1 = re.findall(p1, str_line)
            if str_line1:
                str_line1 = '{' + str_line1[0] + '}'
                dict_line = eval(str_line1)

    
                for k, v in dict_line.items():
                    if k == 'sent':
                        v = v.replace(' ', '')
                        sent.append(v)
                    elif k == 'ners':
                        ner_label.append(v)
                    else:
                        pass
    return sent, ner_label



def data_enhancement(train_dir):
    
    # train_sent, train_ner_label = train_raw_text, train_raw_labels
    train_sent, train_ner_label = get_example_sent_ner(train_dir)

    #! 建立类别实体库
    # 存放idx
    binglifen_qi = [] #病理分期 2
    binglifen_xing = [] # 病理分型    31
    binglifen_ji = [] # 病理分级 39
    qixiang = [] # 期象         137
    leijibuwei = [] # 累及部位      434
    shoushu = [] # 手术        877
    zhidai = [] # 指代        1115
    idx_list = [zhidai, shoushu, qixiang, leijibuwei, binglifen_ji, binglifen_xing, binglifen_qi]
    # 实体库
    ent_binglifen_qi = [] #病理分期 2
    ent_binglifen_xing = [] # 病理分型    31
    ent_binglifen_ji = [] # 病理分级 39
    ent_qixiang = [] # 期象         137
    ent_leijibuwei = [] # 累及部位      434
    ent_shoushu = [] # 手术        877
    ent_zhidai = [] # 指代        1115
    ent_list = [ent_zhidai, ent_shoushu, ent_qixiang, ent_leijibuwei, ent_binglifen_ji, ent_binglifen_xing, ent_binglifen_qi]

    for i, labels in enumerate(train_ner_label):
        for _label in labels:
            if _label[2] == '病理分期' and i not in binglifen_qi:
                binglifen_qi.append(i)
                if _label[3] not in ent_binglifen_qi:
                    ent_binglifen_qi.append(_label[3])

            if _label[2] == '病理分型' and i not in binglifen_xing: 
                binglifen_xing.append(i)
                if _label[3] not in ent_binglifen_xing:
                    ent_binglifen_xing.append(_label[3])
            
            if _label[2] == '病理分级' and  i not in binglifen_ji:
                binglifen_ji.append(i)
                if _label[3] not in ent_binglifen_ji:
                    ent_binglifen_ji.append(_label[3])

            if _label[2] == '期象' and i not in qixiang:
                qixiang.append(i)
                if _label[3] not in ent_qixiang:
                    ent_qixiang.append(_label[3])

            if _label[2] == '累及部位' and i not in leijibuwei:
                leijibuwei.append(i)
                if _label[3] not in ent_leijibuwei:
                    ent_leijibuwei.append(_label[3])

            if _label[2] == '手术' and i not in shoushu:
                shoushu.append(i)
                if _label[3] not in ent_shoushu:
                    ent_shoushu.append(_label[3])
            
            if _label[2] == '指代' and i not in zhidai:
                zhidai.append(i)
                if _label[3] not in ent_zhidai:
                    ent_zhidai.append(_label[3])
                    
    #! 实体随机替换
    # ent_replace = [] # 存放已经使用过的实体
    replaced_labels = [] # 存放替换好的labels
    replaced_sents = [] # 存放替换好的sent
    for i, type in enumerate(ENTITY_TYPES):
        train_sent, train_ner_label = get_example_sent_ner(train_dir)
        # train_sent, train_ner_label = train_raw_text, train_raw_labels
        ent_type = ent_list[i]
        for idx in idx_list[i]:
            ner_label = train_ner_label[idx]
            ner_sent = train_sent[idx]
            for _label_idx, _label in enumerate(ner_label):
                if _label[2] == type:
                    ent = _label[3]

                    ent1 = random.choice(ent_type)
                    while ent1 == ent:
                        ent1 = random.choice(ent_type)
                    # 替换实体
                    # 若前面有token与ent相同 会出现替换错误
                    # ner_sent = ner_sent.replace(ent, ent1, 1)
                    sent_word = ner_sent[_label[0]: ]
                    sent_word = sent_word.replace(ent, ent1, 1)
                    ner_sent = ner_sent[: _label[0]] + sent_word

                    # 如果token长度不同 需要调整后续label
                    if len(ent) != len(ent1):
                        ent_gap = len(ent1) - len(ent) # 两个实体差多少个token
                        # ner_label[_label_idx][1] = ner_label[_label_idx][0] + len(ent1)# end_idx
                        ner_label[_label_idx][1] += ent_gap # end_idx
                        ner_label[_label_idx][3] = ent1
                        assert ner_sent[ner_label[_label_idx][0]: ner_label[_label_idx][1]] == ent1
        
                        # 处理后续label
                        for _label_idx1 in range(_label_idx+1, len(ner_label)):
                            ner_label[_label_idx1][0] += ent_gap
                            ner_label[_label_idx1][1] += ent_gap
                            assert ner_sent[ner_label[_label_idx1][0]: ner_label[_label_idx1][1]] == ner_label[_label_idx1][3]
                    # 若token长度相同只需要调整当前label
                    else:
                        ner_label[_label_idx][3] = ent1
                        
                        
            replaced_sents.append(ner_sent)
            replaced_labels.append(ner_label)
            # if type in ['期象', '累及部位', '病理分级', '病理分型', '病理分期']:
            #     eda_sent, eda_labels = data_enhancence_eda(ner_sent, ner_label)
    return replaced_sents, replaced_labels
# 随机选择句子进行实体替换
# from nlpcda import Similarword             
# def data_enhancence_eda(train_dir):
#     train_sent, train_ner_label = get_example_sent_ner(train_dir)   
    

# train_dir = '/public/ch/project/CH_Entity_Recognize/data/raw_data/train.conll'              
# replaced_sents, replaced_labels = data_enhancement(train_dir)
# print()