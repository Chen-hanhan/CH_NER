import os
import copy
import torch
import logging
from torch.cuda.amp import autocast as ac
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
# from src.utils.attack_train_utils import FGM, PGD
from src.utils.functions_utils import load_model_and_parallel, swa
import wandb
wandb.login()

logger = logging.getLogger(__name__)

def save_model(opt, model, global_step):
    output_dir = os.path.join(opt.output_dir, '{}-checkpoint-{}'.format(opt.version, global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_to_save = (model.module if hasattr(model, 'module') else model)
    logger.info(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')

    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))


def build_optimizer_and_scheduler(opt, model, t_total):
    module = (model.module if hasattr(model, "module") else model)

    # 差分学习率

    no_decay = ['bias', 'LayerNorm.weight']
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
    # for name, para in module.name_parameter():
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))
    #TODO:差分学习率为何如此设计
    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )
    return optimizer, scheduler

def train(opt, model, train_dataset):
    swa_raw_model = copy.deepcopy(model)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=opt.train_batch_size,
                                  num_workers=0,
                                  sampler=train_sampler)


    scaler = None
    if opt.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    model, device = load_model_and_parallel(model, opt.gpu_ids)
    #TODO:bug：无法使用多gpu
    use_n_gpus = False
    if hasattr(model, "module"):
        use_n_gpus = True
    
    t_total = len(train_loader) * opt.train_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)

     # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {opt.train_epochs}")
    logger.info(f"  Total training batch size = {opt.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0

    model.zero_grad()

    save_steps = t_total // opt.train_epochs
    eval_steps = save_steps

    logger.info(f'Save model in {save_steps} steps; Eval model in {eval_steps} steps')

    log_loss_steps = 20

    avg_loss = 0.

    wandb.init(
      # Set entity to specify your username or team name
      # ex: entity="carey",
      # Set the project where this run will be logged
      project="baseline", 
      # Track hyperparameters and run metadata
      config={
      "version": opt.version,
      "bert_learning_rate": opt.lr,
      "other_learning_rate": opt.other_lr,
      "train_batch_size": opt.train_batch_size,
      "epoch": opt.train_epochs,
      "dropout_prob": opt.dropout_prob,
      "architecture": "Bert+CRF",
      "dataset": "3195-train,791-dev",})
    #TODO:tqam
    for epoch in range(opt.train_epochs):

        for step, batch_data in enumerate(train_loader):

            model.train()

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            
            if opt.use_fp16:
                with ac():
                    loss = model(**batch_data)[0]
            else:
                loss = model(**batch_data)[0]

            if use_n_gpus:
                loss = loss.mean()

            if opt.use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if opt.use_fp16:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

            # optimizer.step()
            if opt.use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            model.zero_grad()

            global_step += 1

            if global_step % log_loss_steps == 0:
                avg_loss /= log_loss_steps
                logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
                wandb.log({"avg_loss":avg_loss})
            else:
                avg_loss += loss.item()
                wandb.log({"loss":loss})
                
            if global_step % save_steps == 0:
                save_model(opt, model, global_step)

    # wandb.finish()
    
    swa(swa_raw_model, opt.output_dir, swa_start=opt.swa_start)

    # clear cuda cache to avoid OOM
    torch.cuda.empty_cache()
    logger.info('Train done')

























































