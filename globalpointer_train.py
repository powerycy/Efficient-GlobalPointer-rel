import re
import json
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertModel,AdamW,get_linear_schedule_with_warmup
from data_processing.data_process import yeild_data
from model.model import  EfficientGlobalPointerNet as GlobalPointerNet
from loss_function.loss_fun import multilabel_categorical_crossentropy,global_pointer_crossentropy
from metrics.metrics import global_pointer_f1_score
import sys
import argparse
import torch.distributed as dist
from utils.tools import reduce_tensor
import logging
from tools import setup_seed
from inference_model.inference import NER
from data_process import load_eval,load_data
setup_seed(1234)
# torch.cuda.manual_seed_all(seed)
# from inference import NamedEntityRecognizer
# NER = NamedEntityRecognizer()
# from torch.nn.parallel import DistributedDataParallel as DDP
#DDP
# from torch.utils.data.distributed import DistributedSampler
# 1) 初始化
# parser = argparse.ArgumentParser()
# parser.add_argument('--local_rank', default=-1, type=int,
#                     help='node rank for distributed training')
# args = parser.parse_args()
# local_rank = torch.distributed.get_rank()
# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)
# dist.init_process_group(backend='nccl')
# device = torch.device(f'cuda:{args.local_rank}')
from tqdm import tqdm
#DP
gpus = [4,5,6,7]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("Using {} device".format(device))
import configparser
con = configparser.ConfigParser()
file = './train_config/config.ini'
con.read(file,encoding='utf8')
items = con.items('path')
path = dict(items)
items = con.items('model_superparameter')
model_sp = dict(items)
model_path = path['model_path']
train_file_data = path['train_file_data']
val_file_data = path['val_file_data']
model_save_path = path['model_save_path']
head_size = eval(model_sp['head_size'])
hidden_size = eval(model_sp['hidden_size'])
learning_rate = eval(model_sp['learning_rate'])
clip_norm = eval(model_sp['clip_norm'])
re_maxlen = eval(model_sp['re_maxlen'])
train_dataloader,categories_size,categories2id,id2categories = yeild_data(train_file_data,is_train=True,DDP=False)
val_data = load_eval(train_file_data)
# val_dataloader = yeild_data(val_file_data,categories_size=categories_size,categories2id=categories2id,is_train=False,DDP=False)
model = GlobalPointerNet(model_path,categories_size,head_size,hidden_size,rel)
model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
# model = DDP(model,device_ids=[args.local_rank],find_unused_parameters=True)
epochs = eval(model_sp['epochs'])
warmup_steps = eval(model_sp['warmup_steps'])
total_steps = len(train_dataloader) * epochs
param_optimizer = list(model.named_parameters())
# train_epoch_loss = 0
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
def train(epoch,dataloader, model, loss_func, optimizer):
    model.train()
    size = len(dataloader.dataset)
    numerate, denominator = 0, 0
    numerate_head, denominator_head = 0, 0
    numerate_tail, denominator_tail = 0, 0
    for batch, (data,label,head_label,tail_label) in enumerate(dataloader):
        z = 3 # 2倍的关系
        steps_per_ep = len(dataloader) #有多少数据
        total_steps = 7*steps_per_ep + 1 # + 1 avoid division by zero error #加速loss在一定的步数回归
        current_step = steps_per_ep * epoch + batch # 
        w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z) # 
        w_rel = min((1 / z) * current_step / total_steps, (1 / z)) # 设置首先针对实体训练，然后在抽取关系
        loss_weights = {"ent": w_ent, "rel": w_rel} #给予不同任务的权重
        input_ids = data['input_ids'].squeeze().to(device)
        attention_mask = data['attention_mask'].squeeze().to(device)
        token_type_ids = data['token_type_ids'].squeeze().to(device)
        label = label.to(device)
        head_label = head_label.to(device)
        tail_label = tail_label.to(device)
        pred_ent,pred_head,pred_tail = model(input_ids,attention_mask,token_type_ids)
        loss_ent = loss_func(label,pred_ent)
        loss_head = loss_func(head_label,pred_head)
        loss_tail = loss_func(tail_label,pred_tail)
        temp_n,temp_d = global_pointer_f1_score(label,pred_ent)
        temp_n_head,temp_d_head = global_pointer_f1_score(head_label,pred_head)
        temp_n_tail,temp_d_tail = global_pointer_f1_score(tail_label,pred_tail)
        numerate += temp_n
        denominator += temp_d
        
        numerate_head += temp_n_head
        denominator_head += temp_d_head

        numerate_tail += temp_n_tail
        denominator_tail += temp_d_tail
        # Backpropagation
        w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
        loss = w_ent * loss_ent + w_rel * loss_head + w_rel * loss_tail
        loss.backward()
        # fgm.attack()
        # pred = model(input_ids,attention_mask,token_type_ids,input_ids_type ,attention_mask_type ,token_type_ids_type)
        # loss_adv = loss_func(label,pred,attention_mask,is_train=False)
        # loss_adv.backward()
        # fgm.restore()
        # temp_n,temp_d = global_pointer_f1_score(label,pred)
        numerate += temp_n
        denominator += temp_d
        # loss_adv =  output[0].mean()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Train_ent F1: {(2*numerate/denominator):>2f},Train_head F1:{(2*numerate_head/denominator_head):>2f},Train_tail F1:{(2*numerate_tail/denominator_tail):>2f}")
    # print(f"Train_ent F1: {(2*numerate/denominator):>2f},Train_head F1:{(2*numerate_head/denominator_head):>2f}%")

def evaluate(dataloader,loss_func, model):
    size = len(dataloader.dataset)
    model.eval()
    val_loss = 0
    numerate, denominator = 0, 0
    numerate_head, denominator_head = 0, 0
    numerate_tail, denominator_tail = 0, 0
    with torch.no_grad():
        for data,label,head_label,tail_label in dataloader:
            input_ids = data['input_ids'].squeeze().to(device)
            attention_mask = data['attention_mask'].squeeze().to(device)
            token_type_ids = data['token_type_ids'].squeeze().to(device)
            label = label.to(device)
            head_label = head_label.to(device)
            tail_label = tail_label.to(device)
            pred,pred_head,pred_tail = model(input_ids,attention_mask,token_type_ids)
            val_loss += (loss_func(label,pred).item() + loss_func(head_label,pred_head).item() + loss_func(tail_label,pred_tail).item())

            # val_loss += (loss_func(label,pred).item()  + loss_func(head_label,pred_head).item())
            temp_n,temp_d = global_pointer_f1_score(label,pred)
            temp_n_head,temp_d_head = global_pointer_f1_score(head_label,pred_head)
            temp_n_tail,temp_d_tail = global_pointer_f1_score(tail_label,pred_tail)

            numerate += temp_n
            denominator += temp_d
            

            numerate_head += temp_n_head
            denominator_head += temp_d_head

            numerate_tail += temp_n_tail
            denominator_tail += temp_d_tail
    val_loss /= size
    val_f1_ent = 2*numerate/denominator
    val_f1_head = 2*numerate_head/denominator_head
    val_f1_tail = 2*numerate_tail/denominator_tail
    print(f"Val:\n F1_ent:{(val_f1_ent):>2f},F1_head:{(val_f1_head):>2f},F1_tail:{(val_f1_tail):>2f},Avg loss: {val_loss:>2f} \n")
    # print(f"Val:\n F1_ent:{(val_f1_ent):>2f},F1_head:{(val_f1_head):>2f},Avg loss: {val_loss:>2f} \n")
    return val_f1_ent,val_f1_head,val_f1_tail
def evaluate_val(data,model):
    """评测函数
    """
    X_ent, Y_ent, Z_ent = 1e-10, 1e-10, 1e-10
    X_rel, Y_rel, Z_rel = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R_ent,R_rel = NER.recognize(d[0],id2categories,model)
        R_ent = set(R_ent)
        R_rel = set(R_rel)
        d_rel = d.index('relation')
        T_ent = set([tuple(i) for i in d[1:d_rel]])
        T_rel = set([tuple(i) for i in d[d_rel+1:]])

        X_ent += len(R_ent & T_ent)
        Y_ent += len(R_ent)
        Z_ent += len(T_ent) 

        X_rel += len(R_rel & T_rel)
        Y_rel += len(R_rel)
        Z_rel += len(T_rel)

    f1_ent, precision_ent, recall_ent = 2 * X_ent / (Y_ent + Z_ent), X_ent / Y_ent, X_ent / Z_ent
    f1_rel, precision_rel, recall_rel = 2 * X_rel / (Y_rel + Z_rel), X_rel / Y_rel, X_rel / Z_rel
    return f1_ent, precision_ent, recall_ent,f1_rel, precision_rel, recall_rel

def evaluate_val_1(data,model):
    """评测函数
    """
    X_ent, Y_ent, Z_ent = 1e-10, 1e-10, 1e-10
    X_rel, Y_rel, Z_rel = 1e-10, 1e-10, 1e-10
    X_tail, Y_tail, Z_tail = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R_ent,R_rel,R_tail = NER.recognize(d[0],id2categories,model)
        R_ent = set(R_ent)
        R_rel = set(R_rel)
        R_tail = set(R_tail)
        d_rel = d.index('relation_head')
        d_tail = d.index('relation_tail')

        T_ent = set([tuple(i) for i in d[1:d_rel]])
        T_rel = set([tuple(i) for i in d[d_rel+1:d_tail]])
        T_tail = set([tuple(i) for i in d[d_tail+1:]])

        X_ent += len(R_ent & T_ent)
        Y_ent += len(R_ent)
        Z_ent += len(T_ent) 

        X_rel += len(R_rel & T_rel)
        Y_rel += len(R_rel)
        Z_rel += len(T_rel)

        X_tail += len(R_tail & T_tail)
        Y_tail += len(R_tail)
        Z_tail += len(T_tail)

    f1_ent = 2 * X_ent / (Y_ent + Z_ent)
    f1_rel = 2 * X_rel / (Y_rel + Z_rel)
    f1_tail = 2 * X_tail/ (Y_tail + Z_tail)
    return f1_ent, f1_rel,f1_tail
class Evaluator(object):
    """评估与保存
    """
    def __init__(self,best_val_f1,best_rel_f1):
        self.best_val_f1 = best_val_f1
        # self.best_rel_head_f1 = best_rel_head_f1
        self.best_rel_f1 = best_rel_f1
        # self.best_rel_tail_f1 = best_rel_tail_f1
    def on_epoch_end(self, epoch, logs=None):
        f1_ent, _, _,f1_rel, _, _ = evaluate_val(val_data,model)
        print('f1_ent',f1_ent,'f1_rel',f1_rel)
        # f1_ent, f1_rel, f1_tail = evaluate_val_1(val_data,model)
        # f1_ent,f1_rel,f1_rel_1 = evaluate(val_dataloader,global_pointer_crossentropy, model)
        # 保存最优
        # if f1_ent >= self.best_val_f1  and f1_rel >= self.best_rel_f1:
        #     self.best_val_f1 = f1_ent
        #     self.best_rel_f1 = f1_rel
        #     # self.best_rel_tail_f1 = f1_tails
        #     torch.save(model.module.state_dict(), f=model_save_path)
        # if args.local_rank == 0:
        # print(
            # 'valid:  f1_ent: %.2f,f1_rel: %.2f\n'%
            # (f1_ent,f1_rel)
                    # if args.local_rank == 0:
            # 'valid:  f1_ent: %.2f,  best f1_ent: %.2f,f1_head: %.2f,  best f1_head: %.2f\n'%
            # (f1_ent,self.best_val_f1,f1_head,self.best_rel_head_f1)
            # 'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            # (f1, precision, recall, self.best_val_f1)
        # )
        return self.best_val_f1,self.best_rel_f1
def run_model(optimizer):
    best_val_f1 = 0
    best_rel_f1 = 0
    for epoch in range(epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        print(f"Epoch {epoch + 1}")
        train(epoch,train_dataloader, model, global_pointer_crossentropy, optimizer)
        best_val_f1, best_rel_f1 = Evaluator(best_val_f1,best_rel_f1,).on_epoch_end(epoch)
    print('end')
if __name__ == '__main__':
    run_model(optimizer)