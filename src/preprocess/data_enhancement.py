import re
import numpy as np
import random
import string
import copy
ENTITY_TYPES = ['指代', '手术','期象', '累及部位', '病理分级', '病理分型', '病理分期']
ENTITY_TYPES_PY = ['zhidai', 'shoushu', 'qixiang', 'leijibuwei', 'binglifen_ji', 'binglifen_xing', 'binglifen_qi']
ENTITY = ['ent_zhidai', 'ent_shoushu', 'ent_qixiang', 'ent_leijibuwei', 'ent_binglifen_ji', 'ent_binglifen_xing', 'ent_binglifen_qi']
ALL_ENTITY_TYPES =  ['器官组织', '阴性表现', '属性', '阳性表现', '否定描述', 
                '修饰描述', '异常现象', '数量', '测量值', '检查手段', '疾病',
                 '指代','手术','期象', '累及部位', '病理分级', '病理分型', '病理分期']


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


# 建立类别实体库
def crete_ent_type(train_ner_label):
    
    binglifen_qi = [] #病理分期 2
    binglifen_xing = [] # 病理分型    31
    binglifen_ji = [] # 病理分级 39
    qixiang = [] # 期象         137
    leijibuwei = [] # 累及部位      434
    shoushu = [] # 手术        877
    zhidai = [] # 指代        1115
    idx_list = [zhidai, shoushu, qixiang, leijibuwei, binglifen_ji, binglifen_xing, binglifen_qi]
    # 实体库
    dict_ent_types = {}
    for i in range(len(ALL_ENTITY_TYPES)):
        ent_types = []
        
        for sent_idx, labels in enumerate(train_ner_label):
            for _label in labels:
                if _label[2] == ALL_ENTITY_TYPES[i]:
                    if _label[3] not in ent_types:
                        ent_types.append(_label[3])
                    if 10 < i < 18 and sent_idx not in idx_list[i-11]:
                        idx_list[i-11].append(sent_idx)
        
        dict_ent_types[ALL_ENTITY_TYPES[i]] = ent_types
    # 器官组织，4618，阴性表现 211，属性 342，阳性表现 719，否定描述 6， 修饰描述 476， 异常现象 1127， 数量 56，
    # 测量值 1868，检查手段 226，疾病 911，指代 61，手术 400，期像 11，累及部位 190，病理分级 17，病理分型 19，病理分期 2
    # 需要扩充的实体库： 阴性表现，否定描述，检查手段，指代，期像，累及部位，病理分级，病理分型，病理分期 
    for ent in ['缺损']:
        dict_ent_types['阴性表现'].append(str(ent))
    # dict_ent_types['否定描述'].append([])
    # dict_ent_types['检查手段'].append([])
    # for ent in ['左侧最宽', '中上段', '上段','卫星灶', '边缘']:
    #     dict_ent_types['指代'].append(str(ent))
    
    for ent in ['收缩期','舒张期']:
        dict_ent_types['期象'].append(str(ent))
    
    # dict_ent_types['累及部位'].append(['基底节', '皮质', '脊髓'])
    # for ent in ['C-TIRADS6a','C-TIRADS分级,0', 'C-TIRADS分级,4b', 'C-TIRADS分级，6','A-TIRADS4b']:
    #     dict_ent_types['病理分级'].append(str(ent))
    
    # for ent in ['BI-RADS分类0', 'BI-RADS分类4b', 'C-TIRADS-6类','C-TIRADS分类0', 'C-TIRADS分类4b', 'C-TIRADS分类6']:
    #     dict_ent_types['病理分型'].append(str(ent))
   
    for ent in ['IA期', 'T1期', 'IB期', 'T2期', 'III期']:
        dict_ent_types['病理分期'].append(str(ent))
        
    return idx_list, dict_ent_types


def data_enhancement(train_dir):
    
    # train_sent, train_ner_label = train_raw_text, train_raw_labels
    train_sent, train_ner_label = get_example_sent_ner(train_dir)
    # 建立类别实体库
    idx_list, dict_ent_types = crete_ent_type(train_ner_label)
    
    #! 实体随机替换
    # ent_replace = [] # 存放已经使用过的实体
    replaced_labels = [] # 存放替换好的labels
    replaced_sents = [] # 存放替换好的sent
    for i, type in enumerate(ENTITY_TYPES):
        train_sent, train_ner_label = get_example_sent_ner(train_dir)
        # train_sent, train_ner_label = train_raw_text, train_raw_labels
        ent_type = dict_ent_types[type]
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
            if type in ['期象', '累及部位', '病理分级', '病理分型', '病理分期']:               
                eda_sent, eda_label = data_enhancence_1(ner_sent, ner_label, dict_ent_types)
                for x in eda_sent:
                    replaced_sents.append(x)
                for x in eda_label:
                    replaced_labels.append(x)
                
    return replaced_sents, replaced_labels


def data_enhancence_1(ner_sent, ner_label, dict_ent_types):
    out_ner_sent = []
    out_ner_labels = []
    for _ in range(5):
        ner_sent1 = ner_sent
        ner_label1 = copy.deepcopy(ner_label)
        for _label_idx, _label in enumerate(ner_label1):
            type = _label[2]
            ent = _label[3]
            ent_type = dict_ent_types[type]
            ent1 = random.choice(ent_type)
            while ent1 == ent:
                ent1 = random.choice(ent_type)
            
            # 替换实体       
            sent_word = ner_sent1[_label[0]: ]
            sent_word = sent_word.replace(ent, ent1, 1)
            ner_sent1 = ner_sent1[: _label[0]] + sent_word
            
            if len(ent) != len(ent1):
                
                ent_gap = len(ent1) - len(ent) # 两个实体差多少个token
                
                ner_label1[_label_idx][1] += ent_gap # end_idx
                ner_label1[_label_idx][3] = ent1
                assert ner_sent1[ner_label1[_label_idx][0]: ner_label1[_label_idx][1]] == ent1

                # 处理后续label
                for _label_idx1 in range(_label_idx+1, len(ner_label1)):
                    ner_label1[_label_idx1][0] += ent_gap
                    ner_label1[_label_idx1][1] += ent_gap
                    assert ner_sent1[ner_label1[_label_idx1][0]: ner_label1[_label_idx1][1]] == ner_label1[_label_idx1][3]
                    
            # 若token长度相同只需要调整当前label
            else:
                ner_label1[_label_idx][3] = ent1
            
        out_ner_sent.append(ner_sent1)
        out_ner_labels.append(ner_label1)
    
    return  out_ner_sent, out_ner_labels 
            
            




# 随机选择句子进行实体替换
# from nlpcda import Similarword, Homophone  
# smw = Similarword(create_num=2, change_rate=0.5)# 同义词
# smw_hp = Homophone(create_num=2, change_rate=0.3)# 近义词 
# 同义词 近义词替换     
def data_enhancence_eda(ner_sent, ner_label):
 

    # for _label_idx, _label in enumerate(ner_label):
    label_idx = 0
    eda_sent = ''
    while label_idx < (len(ner_label)-1):
        if label_idx == 0:# 第一个实体
            text0 = ner_sent[: ner_label[label_idx][0]]
            if text0:
                tmp_text0 = smw.replace(text0)# 返回列表
                
                if len(tmp_text0) < 2:
                    tmp_text0 = tmp_text0[0]
                else:
                    tmp_text0 = tmp_text0[1]
                
                # 如果相等就取近义词
                if tmp_text0 == text0:
                    if len(tmp_text0) < 2:
                        tmp_text0 = tmp_text0[0]
                    else:
                        tmp_text0 = tmp_text0[1]
                # 如果长度不等则进行处理
                while len(tmp_text0) != len(text0):
                    # 长了则随机删除
                    if len(tmp_text0) > len(text0):
                        
                        text0_gap = len(tmp_text0) - len(text0)
                        for _ in range(text0_gap):
                            tmp_text0 = tmp_text0.replace(random.choice(tmp_text0), '', 1)
                    
                    else:
                        tmp_text0 = text0
                    
                eda_sent += tmp_text0
        
        text1 = ner_sent[ner_label[label_idx][1]: ner_label[label_idx+1][0]]
        if text1:
            tmp_text1 = smw.replace(text1)# 太短的情况下有可能返回一个
            if len(tmp_text1) < 2:
                tmp_text1 = tmp_text1[0]
                
            else:
                tmp_text1 = tmp_text1[1]
            # 如果相等就取近义词 # 如果为标点符号
            if tmp_text1 == text1:
                tmp_text1 = smw_hp.replace(tmp_text1)
                if len(tmp_text1) < 2:
                    tmp_text1 = tmp_text1[0]
                else:
                    tmp_text1 = tmp_text1[1]
                
            while len(tmp_text1) != len(text1):
                # 长了则随机删除
                if len(tmp_text1) > len(text1):
                    
                    text1_gap = len(tmp_text1) - len(text1)
                    for _ in range(text1_gap):
                        tmp_text1 = tmp_text1.replace(random.choice(tmp_text1), '', 1)
                
                else:
                    tmp_text1 = text1
                    
            eda_sent += (ner_label[label_idx][3] + tmp_text1)
                
        else:
            eda_sent += ner_label[label_idx][3]         
            
        if (label_idx+2) == len(ner_label):
            eda_sent += ner_label[label_idx+1][3]   
                
        assert eda_sent[ner_label[label_idx][0]: ner_label[label_idx][1]] == ner_label[label_idx][3]  
        label_idx += 1

    
    
    return eda_sent
    



# train_dir = '/public/ch/project/CH_Entity_Recognize/data/raw_data/train.conll'              
# replaced_sents, replaced_labels = data_enhancement(train_dir)
# print()