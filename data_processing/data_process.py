from re import search
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizerFast
import json
from utils.tools import token_rematch
import configparser
con = configparser.ConfigParser()
file = './train_config/config.ini'
con.read(file,encoding='utf8')
items = con.items('path')
path = dict(items)
items = con.items('model_superparameter')
model_sp = dict(items)
model_path = path['model_path']
maxlen = eval(model_sp['maxlen'])
batch_size = eval(model_sp['batch_size'])
from torch.utils.data.distributed import DistributedSampler
tokenizer = BertTokenizerFast.from_pretrained(model_path,do_lower_case= True)
import json
categories = ['期象', '累及部位', '否定描述', '修饰描述', '病理分级', '数量', '病理分期', '疾病', '指代', '阳性表现', '测量值', '手术', '属性', '检查手段', '阴性表现','异常现象', '器官组织','病理分型']
def load_data(filename,is_train):
    
    resultList = []
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    # categories = ['期象', '累及部位', '否定描述', '修饰描述', '病理分级', '数量', '病理分期', '疾病', '指代', '阳性表现', '测量值', '手术', '属性', '检查手段', '阴性表现','异常现象', '器官组织','病理分型']
    D = []
    for d in json.load(open(filename)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_id'], e['end_id'], e['type']
            if start <= end:
                D[-1].append((start, end, label))
            resultList.append(label)
        D[-1].append('relation_head')
        for r in d['relation_list']:
            start_head, end_head, predicate = r['subobj_head_start_idx'], r['subobj_head_end_idx'], r['predicate']
            if start_head <= end_head:
                D[-1].append((start_head, end_head, predicate))
        D[-1].append('relation_tail')
        for r in d['relation_list']:
            start_tail, end_tail, predicate = r['subobj_tail_start_idx'], r['subobj_tail_end_idx'], r['predicate']
            # if start_tail <= end_tail:
            D[-1].append((start_tail, end_tail, predicate))
    # categories = list(set(resultList))
    # categories.sort(key=resultList.index)
    if is_train:
        return D,categories
    else:
        return D
def load_eval(filename):
    D = []
    for d in json.load(open(filename)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_id'], e['end_id'], e['type']
            if start <= end:
                D[-1].append((start, end, label))
        D[-1].append('relation')    
        for r in d['relation_list']:
            sub_start, sub_end, predicate = r['subject_start_id'], r['subject_end_id'], r['predicate']
            obj_start, obj_end = r['object_start_id'], r['object_end_id']
            if sub_start <= sub_end and obj_start <= obj_end:
                D[-1].append((sub_start, sub_end,obj_start,obj_end,predicate)) 
    return D
class NerDataset(Dataset):
    def __init__(self, data, tokenizer,categories,categories_size,categories2id,categories_rel,categories2id_rel):
        self.data = data
        self.categories = categories
        # self.max_type = 59
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.categories_size = categories_size
        self.categories2id = categories2id
        self.categories_rel = categories_rel
        self.categories2id_rel = categories2id_rel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        label = torch.zeros((self.categories_size,self.maxlen,self.maxlen))
        # head_rel_label = torch.zeros((self.categories_rel,self.maxlen,self.maxlen))
        tail_rel_labe = torch.zeros((self.categories_rel,self.maxlen,self.maxlen))
        context = tokenizer(d[0],return_offsets_mapping=True,max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        # context_ent = tokenizer(''.join(self.categories),return_offsets_mapping=True,max_length=self.max_type,truncation=True,padding='max_length',return_tensors='pt')
        tokens = tokenizer.tokenize(d[0],max_length=self.maxlen,add_special_tokens=True)
        mapping = token_rematch().rematch(d[0],tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        # rel_index_head = d.index('relation_head')
        rel_index_tail = d.index('relation_tail')
        for entity_input in d[1:rel_index_tail]:
            start, end = entity_input[0], entity_input[1]
            if start in start_mapping and end in end_mapping and start < self.maxlen and end < self.maxlen:
                start = start_mapping[start]
                end = end_mapping[end]
                label[self.categories2id[entity_input[2]],start,end] = 1
        # for rel_input_head in d[rel_index_head+1:rel_index_tail]:
        #     start_head, end_head = rel_input_head[0], rel_input_head[1]
        #     if start_head in start_mapping and end_head in end_mapping and start_head < self.maxlen and end_head < self.maxlen:
        #         start_head = start_mapping[start_head]
        #         end_head = end_mapping[end_head]
        #         head_rel_label[self.categories2id_rel[rel_input_head[2]],start_head,end_head] = 1
        for rel_input_tail in d[rel_index_tail+1:]:
            start_tail, end_tail  = rel_input_tail[0], rel_input_tail[1]
            if start_tail in start_mapping and end_tail in end_mapping and start_tail < self.maxlen and end < self.maxlen:
                start_tail = start_mapping[start_tail]
                end_tail = end_mapping[end_tail]
                tail_rel_labe[self.categories2id_rel[rel_input_tail[2]],start_tail,end_tail] = 1
        return context,label,tail_rel_labe
def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    # batch = list(zip(*batch))
    # data =  torch.tensor(batch[0])
    # a = [item[0]['input_ids'] for item in batch]
    # b = torch.cat(a,dim=0)
    # context = torch.cat([item[0]['input_ids'] for item in batch],dim=0)
    text_dict = {}
    # for nums in range(2):
    input_ids = torch.cat([item[0]['input_ids'] for item in batch],dim=0)
    attention_mask = torch.cat([item[0]['attention_mask'] for item in batch],dim=0)    
    token_type_ids = torch.cat([item[0]['token_type_ids'] for item in batch],dim=0)
    label = torch.stack([item[1] for item in batch],dim=0)
    label = torch.cat((label,label),dim=0)
    text_dict['input_ids'] = torch.cat((input_ids,input_ids),dim=0)
    text_dict['attention_mask'] = torch.cat((attention_mask,attention_mask),dim=0)
    text_dict['token_type_ids'] = torch.cat((token_type_ids,token_type_ids),dim=0)
    # label = batch[1]
    # res_type = batch[2]
    # del batch
    # return input_ids,attention_mask,token_type_ids,label,res_type
    return text_dict,label
def yeild_data(train_file_data,is_train,categories = None,categories_size=None,categories2id=None,DDP=True):
    categories2id_rel = {'属性':0}
    # id2categories_rel = {0:'属性'}
    categories_rel = 1
    if is_train:
        train_data, categories = load_data(train_file_data,is_train=is_train)
        categories_size = len(categories)
        categories2id = {c:idx for idx,c in enumerate(categories)}
        id2categories = {idx : c for idx,c in enumerate(categories)}
        train_data = NerDataset(train_data,tokenizer,categories,
        categories_size,categories2id,categories_rel,categories2id_rel)
        if DDP:
            train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, batch_size=batch_size,sampler=train_sampler,shuffle=False)
        else:
            train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
        return train_dataloader,categories_size,categories2id,id2categories
    else:
        train_data = load_data(train_file_data,is_train=is_train)
        train_data = NerDataset(train_data,tokenizer,categories,categories_size,
        categories2id,categories_rel,categories2id_rel)
        if DDP:
            train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, batch_size=batch_size,sampler=train_sampler)
        else:
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
        return train_dataloader

# if __name__ == '__main__':
    # train_data, categories = load_data('/home/yuanchaoyi/DeepKg/ExtractionEntities_xunfei/data/train_data_all.json',is_train=True)
    # a = 1
    # train_data, categories = load_data_rel_head('/home/yuanchaoyi/DeepKg/ExtractionEntities_xunfei/data/train_data.json',is_train=True)
#     # val_data = load_data('/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_dev.json',is_train=False)
#     # categories_size = len(categories)
#     # categories2id = {c:idx for idx,c in enumerate(categories)}
#     # id2categories = {idx : c for idx,c in enumerate(categories)}
    # train_dataloader,val_dataloader,categories_size = yeild_data('/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_train.json','/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_dev.json')
#     a = 1
    # train_list,dev_list = read_file('/home/yuanchaoyi/TPlinker-joint-extraction/ori_data/xunfei_relation/train/train.conll_convert.conll')
    # json.dump(train_list, open('/home/yuanchaoyi/TPlinker-joint-extraction/ori_data/xunfei_data/train_data.json', "w", encoding = "utf-8"), ensure_ascii = False)
    # json.dump(dev_list, open('/home/yuanchaoyi/TPlinker-joint-extraction/ori_data/xunfei_data/dev_data.json', "w", encoding = "utf-8"), ensure_ascii = False)