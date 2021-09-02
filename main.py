from platform import processor
import time
import os
import logging
import json
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from src.utils.trainer import train
from src.utils.options import Args
from src.utils.model_utils import build_model
from src.utils.dataset_utils import NERDataset
# from src.utils.model_utils
from src.utils.evaluator import crf_evaluation
from src.utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel, get_time_dif
from src.preprocess.processor import NERProcessor, convert_examples_to_features
import wandb

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def train_base(opt, train_examples, dev_examples):
    
    with open(os.path.join(opt.mid_data_dir, f'{opt.task_type}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)
    #TODO:为什么只取[0]? 因为返回(CRFFeatures, (callback_info, type_weight))
    train_features = convert_examples_to_features(opt.task_type,
                                                  train_examples,
                                                  opt.max_seq_len,
                                                  opt.bert_dir,
                                                  ent2id)[0]

    train_dataset = NERDataset(opt.task_type, train_features,
                               'train', use_type_embed=opt.use_type_embed)

    if opt.task_type == 'crf':
        model = build_model('crf', opt.bert_dir, num_tags=len(ent2id),
                           dropout_prob=opt.dropout_prob)

    train(opt, model, train_dataset)
    
    if dev_examples is not None:
        
        dev_features, dev_callback_info = convert_examples_to_features(opt.task_type, dev_examples, opt.max_seq_len,
                                                                       opt.bert_dir, ent2id)

        dev_dataset = NERDataset(opt.task_type, dev_features, 'dev', use_type_embed=opt.use_type_embed)

        dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size, shuffle=False, num_workers=0)
        
        dev_info = (dev_loader, dev_callback_info)
        
        model_path_list = get_model_path_list(opt.output_dir)
        
        metric_str = ''

        max_f1 = 0.
        max_f1_step = 0

        max_f1_path = ''
        
        for idx, model_path in enumerate(model_path_list):

            tmp_step = model_path.split('/')[-2].split('-')[-1]

            model, device = load_model_and_parallel(model, opt.gpu_ids[0],
                                                    ckpt_path=model_path)
        
            if opt.task_type == 'crf':
                tmp_metric_str, tmp_f1 = crf_evaluation(model, dev_info, device, ent2id)
            else:
                pass
            
            logger.info(f'In step {tmp_step}:\n {tmp_metric_str}')

            metric_str += f'In step {tmp_step}:\n {tmp_metric_str}' + '\n\n'

            wandb.log({"f1":tmp_f1, 'precision': tmp_metric_str[0], 'recall': tmp_metric_str[1]})
            
            if tmp_f1 > max_f1:
                max_f1 = tmp_f1
                max_f1_step = tmp_step
                max_f1_path = model_path

        # wandb.finish()
        
        max_metric_str = f'Max f1 is: {max_f1}, in step {max_f1_step}'

        logger.info(max_metric_str)

        metric_str += max_metric_str + '\n'

        eval_save_path = os.path.join(opt.output_dir, 'eval_metric.txt')

        with open(eval_save_path, 'a', encoding='utf-8') as f1:
            f1.write(metric_str)
        #TODO:根据版本存放
        with open(os.path.join(opt.output_dir, 'best_ckpt_path.txt'), 'a', encoding='utf-8') as f2:
            f2.write(max_f1_path + '\n')

        del_dir_list = [os.path.join(opt.output_dir, path.split('/')[-2])
                        for path in model_path_list if path != max_f1_path]

        import shutil
        for x in del_dir_list:
            shutil.rmtree(x)
            logger.info('{}已删除'.format(x))
        
        wandb.finish()
    
        
        
        
def training(opt):
    if args.task_type == 'crf':
        processor = NERProcessor(opt.max_seq_len)

    train_raw_text, train_raw_labels = processor.read_data(os.path.join(opt.raw_data_dir, 'train.conll'))

    train_examples = processor.get_example(raw_text=train_raw_text, 
                                           raw_labels=train_raw_labels,
                                           set_type='train')
    #TODO:分数据集
    train_examples, dev_examples = train_test_split(train_examples, test_size=0.2, shuffle=True)
    
    dev_raw_text, dev_raw_labels = [], []
    for example in dev_examples:
        dev_raw_text.append(example.text)
        dev_raw_labels.append(example.labels)
    assert len(dev_raw_text) == len(dev_raw_labels)
    dev_examples = processor.get_example(dev_raw_text, dev_raw_labels, 'dev')
    # dev_examples = None
    # if opt.eval_model:
    #     dev_raw_text, dev_raw_labels = processor.read_data(os.path.join(opt.raw_data_dir, 'dev.conll'))
    #     dev_examples = processor.get_example(dev_raw_text, dev_raw_labels, 'dev')

    train_base(opt, train_examples, dev_examples)


if __name__ == '__main__':
    start_time = time.time()
    logging.info('----------------开始计时----------------')
    logging.info('----------------------------------------')

    args = Args().get_parser()

    assert args.mode in ['train']
    assert args.task_type in ['crf']

    

    set_seed(args.seed)
    # output_dir + bert_type + task_type + 
    args.output_dir = os.path.join(args.output_dir, args.bert_type)
    args.output_dir += f'_{args.task_type}' 
    args.output_dir += f'_{args.version}'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir , exist_ok=True)

    logger.info(f'{args.mode} {args.task_type} in max_seq_len {args.max_seq_len}')

    if args.mode == 'train':
        training(args)

    time_dif = get_time_dif(start_time)
    logging.info("----------本次容器运行时长：{}-----------".format(time_dif))
