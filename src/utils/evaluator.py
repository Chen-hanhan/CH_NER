import torch
import logging
import numpy as np
from collections import defaultdict
from src.preprocess.processor import ENTITY_TYPES


logger = logging.getLogger(__name__)

def get_base_out(model, loader, device):
    """
    每一个任务的 forward 都一样，封装起来
    """
    model.eval()

    with torch.no_grad():
        for idx, _batch in enumerate(loader):

            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)

            tmp_out = model(**_batch)

            yield tmp_out


def crf_decode(decode_tokens, raw_text, id2ent):
    """
    CRF 解码，用于解码 time loc 的提取
    """
    predict_entities = {}

    decode_tokens = decode_tokens[1:-1]  # 除去 CLS SEP token

    index_ = 0

    while index_ < len(decode_tokens):

        token_label = id2ent[decode_tokens[index_]].split('-')
        #TODO:修改为BIO标注
        # BIOES
        # if token_label[0].startswith('S'):
        #     token_type = token_label[1]
        #     tmp_ent = raw_text[index_]

        #     if token_type not in predict_entities:
        #         predict_entities[token_type] = [(tmp_ent, index_)]
        #     else:
        #         predict_entities[token_type].append((tmp_ent, int(index_)))

        #     index_ += 1

        # elif token_label[0].startswith('B'):
        #     token_type = token_label[1]
        #     start_index = index_
            
        #     index_ += 1
        #     while index_ < len(decode_tokens):
        #         temp_token_label = id2ent[decode_tokens[index_]].split('-')

        #         if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
        #             index_ += 1
        #         elif temp_token_label[0].startswith('E') and token_type == temp_token_label[1]:
        #             end_index = index_
        #             index_ += 1

        #             tmp_ent = raw_text[start_index: end_index + 1]

        #             if token_type not in predict_entities:
        #                 predict_entities[token_type] = [(tmp_ent, start_index)]
        #             else:
        #                 predict_entities[token_type].append((tmp_ent, int(start_index)))

        #             break
        #         else:
        #             break
        # else:
        #     index_ += 1
        
        # BIO
        if token_label[0].startswith('B'):
            token_type = token_label[1]
            start_index = index_
            
            index_ += 1
            while index_ < len(decode_tokens):
                temp_token_label = id2ent[decode_tokens[index_]].split('-')

                if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
                    index_ += 1 
                    # 判断下一个token是否还是当前实体:当前token标签不等于下一token标签 or 下一token标签为0
                    if index_<len(decode_tokens):
                    
                        if (decode_tokens[index_-1] != decode_tokens[index_]) or  (decode_tokens[index_] == 0):
                            # 标签范围左闭右开
                            end_index = index_
                            
                            tmp_ent = raw_text[start_index: end_index]
                            if token_type not in predict_entities:
                                predict_entities[token_type] = [(tmp_ent, start_index)]
                            else:
                                predict_entities[token_type].append((tmp_ent, int(start_index)))

                            break
                    # 若为最后一个token了，则该token为end
                    else: 
                        end_index = index_
                        tmp_ent = raw_text[start_index: end_index]
                        
                        if token_type not in predict_entities:
                            predict_entities[token_type] = [(tmp_ent, start_index)]
                        else:
                            predict_entities[token_type].append((tmp_ent, int(start_index)))

                        break
                # 如果B-后为o或者另外的实体类别
                else:
                    # index_ += 1
                    end_index = index_
                    tmp_ent = raw_text[start_index: end_index]
                    
                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(tmp_ent, start_index)]
                    else:
                        predict_entities[token_type].append((tmp_ent, int(start_index)))

                    break
        else:
            index_ += 1
                    
                        
    
    return predict_entities


def calculate_metric(gt, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def crf_evaluation(model, dev_info, device, ent2id):
    dev_loader, (dev_callback_info, type_weight) = dev_info

    pred_tokens = []

    for tmp_pred in get_base_out(model, dev_loader, device):
        pred_tokens.extend(tmp_pred[0])

    assert len(pred_tokens) == len(dev_callback_info)

    id2ent = {ent2id[key]: key for key in ent2id.keys()}

    role_metric = np.zeros([len(ENTITY_TYPES), 3])

    mirco_metrics = np.zeros(3)

    for tmp_tokens, tmp_callback in zip(pred_tokens, dev_callback_info):

        text, gt_entities = tmp_callback

        tmp_metric = np.zeros([len(ENTITY_TYPES), 3])

        pred_entities = crf_decode(tmp_tokens, text, id2ent)

        for idx, _type in enumerate(ENTITY_TYPES):
            if _type not in pred_entities:
                pred_entities[_type] = []

            tmp_metric[idx] += calculate_metric(gt_entities[_type], pred_entities[_type])

        role_metric += tmp_metric

    for idx, _type in enumerate(ENTITY_TYPES):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                 f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]

    









