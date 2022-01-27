# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# gpus = [5,6,7]
import torch
from transformers import BertTokenizerFast
from utils.tools import token_rematch
import numpy as np
from model.model import  EfficientGlobalPointerNet as GlobalPointerNet
from data_processing.data_process import yeild_data,load_data
import json
from torch import nn
# torch.cuda.set_device('cuda:{}'.format(gpus[0]))
from tqdm import tqdm
import configparser
import re
con = configparser.ConfigParser()
file = './train_config/config.iniconfig.ini'
con.read(file,encoding='utf8')
items = con.items('path')
path = dict(items)
items = con.items('model_superparameter')
model_sp = dict(items)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = path['model_path']
model_save_path = path['model_save_path']
train_file_data = path['train_file_data']
test_file_data = path['test_file_data']
out_file = path['out_file']
max_length = eval(model_sp['inference_maxlen'])
head_size = eval(model_sp['head_size'])
hidden_size = eval(model_sp['hidden_size'])
tokenizer = BertTokenizerFast.from_pretrained(model_path)
# train_data, categories = load_data(train_file_data,is_train=True)
_,categories_size,_,id2categories = yeild_data(train_file_data,is_train=True,DDP=False)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def get_mapping(text):
    text_token = tokenizer.tokenize(text,max_length=max_length,add_special_tokens=True)
    # text_token =  text_token
    text_mapping = token_rematch().rematch(text,text_token)
    return text_mapping
class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text,id2categories,categories,model,threshold=0):
       
        mapping = get_mapping(text)
        # text_res = ''.join(text_res)
        # mapping = torch.tensor([mapping])
        encode_dict = tokenizer(text,return_offsets_mapping=True,max_length=max_length,truncation=True,return_tensors='pt')
        data_type = tokenizer(categories,return_offsets_mapping=True,max_length=59,truncation=True,return_tensors='pt')
        input_ids = encode_dict['input_ids'].to(device)
        token_type_ids = encode_dict['token_type_ids'].to(device)
        attention_mask = encode_dict['attention_mask'].to(device)
        input_ids_type = data_type['input_ids'].to(device)
        attention_mask_type = data_type['attention_mask'].to(device)
        token_type_ids_type = data_type['token_type_ids'].to(device)
        scores_ent,scores_tail = model(input_ids,attention_mask,token_type_ids,input_ids_type,attention_mask_type,token_type_ids_type)
        scores_ent,scores_tail = scores_ent[0],scores_tail[0]
        scores_ent[:, [0, -1]] -= np.inf
        scores_ent[:, :, [0, -1]] -= np.inf

        # scores_head[:, [0, -1]] -= np.inf
        # scores_head[:, :, [0, -1]] -= np.inf

        scores_tail[:, [0, -1]] -= np.inf
        scores_tail[:, :, [0, -1]] -= np.inf
        
        entities = []

        entities_dic = {}
        rel_list = []
        # threshold = torch.tensor(threshold).to(device)
        scores_ent = scores_ent.detach().cpu().numpy()
        # scores_head = scores_head.detach().cpu().numpy()
        scores_tail = scores_tail.detach().cpu().numpy()
        for l, start, end in zip(*np.where(scores_ent > threshold)):
            if start < len(mapping) and end < len(mapping):
                if (len(mapping[start]) and len(mapping[end])) > 0:
                    entities.append(
                        # (mapping[start][0], mapping[end][-1], id2categories[l],text[mapping[start][0]:mapping[end][-1]]+1)
                        (mapping[start][0], mapping[end][-1], id2categories[l])
                        
                        # (offset_mapping[start][0], offset_mapping[end][-1], id2categories[l])
                        )
                    if text[mapping[start][0]] in entities_dic:
                        entities_dic[text[mapping[start][0]]].append(text[mapping[start][0]:mapping[end][-1] + 1])
                    else:
                        entities_dic[text[mapping[start][0]]] = [text[mapping[start][0]:mapping[end][-1] + 1]]

                    if text[mapping[end][-1]] in entities_dic:
                        entities_dic[mapping[end][-1]].append((mapping[start][0],mapping[end][-1]))
                    else:
                        entities_dic[mapping[end][-1]] = [(mapping[start][0],mapping[end][-1])]


        for _,start,end in zip(*np.where(scores_tail > threshold)):
            if start < len(mapping) and end < len(mapping):
                if (len(mapping[start]) and len(mapping[end])) > 0:
                    if mapping[start][0] in entities_dic and mapping[end][-1] in entities_dic: 
                        for sub in entities_dic[mapping[start][0]]:
                            for obj in entities_dic[mapping[end][-1]]:
                                rel_list.append((sub[0],sub[1],obj[0],obj[1],'属性'))
        return entities,rel_list
    def recognize_inference(self, text,id2categories,categories,model,threshold=0):
       
        mapping = get_mapping(text)

        # text_res = ''.join(text_res)
        # mapping = torch.tensor([mapping])
        categories = ''.join(categories)
        encode_dict = tokenizer(text,return_offsets_mapping=True,max_length=max_length,truncation=True,return_tensors='pt')
        input_ids = encode_dict['input_ids'].to(device)
        token_type_ids = encode_dict['token_type_ids'].to(device)
        attention_mask = encode_dict['attention_mask'].to(device)
        data_type = tokenizer(categories,return_offsets_mapping=True,max_length=59,truncation=True,return_tensors='pt')
        input_ids_type = data_type['input_ids'].to(device)
        attention_mask_type = data_type['attention_mask'].to(device)
        token_type_ids_type = data_type['token_type_ids'].to(device)
        scores_ent,scores_tail = model(input_ids,attention_mask,token_type_ids,input_ids_type,attention_mask_type,token_type_ids_type)
        scores_ent,scores_tail = scores_ent[0],scores_tail[0]
        scores_ent[:, [0, -1]] -= np.inf
        scores_ent[:, :, [0, -1]] -= np.inf

        # scores_head[:, [0, -1]] -= np.inf
        # scores_head[:, :, [0, -1]] -= np.inf

        scores_tail[:, [0, -1]] -= np.inf
        scores_tail[:, :, [0, -1]] -= np.inf
        
        entities = set()

        entities_dic = {}
        rel_list = set()
        # threshold = torch.tensor(threshold).to(device)
        scores_ent = scores_ent.detach().cpu().numpy()
        # scores_head = scores_head.detach().cpu().numpy()
        scores_tail = scores_tail.detach().cpu().numpy()
        for l, start, end in zip(*np.where(scores_ent > threshold)):
            if start < len(mapping) and end < len(mapping):
                if (len(mapping[start]) and len(mapping[end])) > 0:
                    entities.add(
                     (mapping[start][0], mapping[end][-1]+1, id2categories[l],text[mapping[start][0]:mapping[end][-1]+1])
                        # (mapping[start][0], mapping[end][-1], id2categories[l])
                        
                        # (offset_mapping[start][0], offset_mapping[end][-1], id2categories[l])
                        )
                    if text[mapping[start][0]] in entities_dic:
                        entities_dic[text[mapping[start][0]]].append(text[mapping[start][0]:mapping[end][-1] + 1])
                    else:
                        entities_dic[text[mapping[start][0]]] = [text[mapping[start][0]:mapping[end][-1] + 1]]

                    if text[mapping[end][-1]] in entities_dic:
                        entities_dic[mapping[end][-1]].append((mapping[start][0],mapping[end][-1],text[mapping[start][0]:mapping[end][-1] + 1],id2categories[l]))
                    else:
                        entities_dic[mapping[end][-1]] = [(mapping[start][0],mapping[end][-1],text[mapping[start][0]:mapping[end][-1] + 1],id2categories[l])]


        for _,start,end in zip(*np.where(scores_tail > threshold)):
            if start < len(mapping) and end < len(mapping):
                if (len(mapping[start]) and len(mapping[end])) > 0:
                    if mapping[start][0] in entities_dic and mapping[end][-1] in entities_dic:
                        for sub in entities_dic[mapping[start][0]]:
                            for obj in entities_dic[mapping[end][-1]]:
                                rel_list.add((sub[0],sub[1]+1,sub[3],sub[2],obj[0],obj[1]+1,obj[3],obj[2],'属性'))
        return entities,rel_list

NER = NamedEntityRecognizer()
def predict_to_file(in_file, out_file,categories_size,id2categories):
    """预测到文件
    可以提交到 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
    """
    model = GlobalPointerNet(model_path,categories_size,head_size,hidden_size,rel)
    # model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
    model.load_state_dict(torch.load(model_save_path))
    # data = json.load(open(in_file)
    with open(in_file,'r',encoding='utf8') as data:
        with open(out_file,'w',encoding='utf8') as res_file:
            categories = ['期象', '累及部位', '否定描述', '修饰描述', '病理分级', '数量', '病理分期', 
            '疾病', '指代', '阳性表现', '测量值', '手术', '属性', '检查手段', '阴性表现','异常现象', '器官组织','病理分型']
            for d in tqdm(data, ncols=100):
                d_res = {}
                d_ent = []
                d = re.sub('\n','',d)
                text = re.sub(' ','',d)
                d_res['sent'] = d 
                entities,rel_list = NER.recognize_inference(text,id2categories,categories,model)
                entities_1 = entities
                for e in entities:
                    d_ent.append([
                            e[0],
                            e[1],
                            e[2],
                            e[3]
                    ])
                    
                d_res['ners'] = d_ent
                res_file.write(json.dumps(d_res,ensure_ascii=False)+'\n')
                for r in rel_list:
                    # if r[0] > r[4]:
                    #     res_file.write(str(r[4:8]) +' '+ '[]'+' '+'属性' +'\n')
                    if tuple(r[:4]) in entities_1:
                        entities_1.remove(r[:4])
                    if tuple(r[4:]) in entities_1:
                        entities_1.remove(r[4:])
                    res_file.write(str(list(r[:4])) + '\t' + str(list(r[4:8])) + '\t' + '属性' +'\n')
                for e in entities_1:
                    res_file.write(str(list(e)) + '\t' + str(list())+'\t'+'属性' +'\n')
                res_file.write('\n')
if __name__ == "__main__":
    predict_to_file(test_file_data,out_file,categories_size,id2categories)
