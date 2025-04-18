import random
import torch
import torch.nn as nn

import argparse
import torch
import sys
import os
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from dataset import Generate_Dataset
import torch.nn.functional as F
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
sys.path.append(os.getcwd())

from models.model import finetune_Model2

#配置logger
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def mark_only_bias_as_trainable(model, args):
    canshu = []
    for i in args.add_bias_layer_idx:
        bias = 'bias_{}'.format(i)
        layer = 'model.layers.{}.'.format(i + 1)
        canshu.append(bias)
        canshu.append(layer + 'self_attn.o_proj.weight')  
        canshu.append(layer + 'mlp.down_proj.weight') 
    
    for n, p in model.named_parameters():
        if n in canshu:    
            p.requires_grad = True
        else:
            p.requires_grad = False
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print(n)

def train_eval(train_dataloader, eval_dataloader, model, optimizer, epoch, logger):
    model.model.train()
    i = 0
    all_loss = 0
    for d in tqdm(train_dataloader,desc="TRAIN/EPOCH{:02d}".format(epoch+1)):
        i += 1
        texts = d['texts']
        labels = d['labels']
        choices = d['choices']

        loss = model.forward(texts, labels, choices)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss += loss.item()
        logger.info('epoch: {0:02d}, batch: {1:03d}, loss: {2:.4f}'.format(epoch+1, i, loss))
    logger.info('##TRAIN## epoch: {0:02d}, avg_loss: {1:.4f}'.format(epoch+1, all_loss/i))

    avg_f1 = eval(eval_dataloader, model, epoch, logger,'VAL')
    return avg_f1

def eval(dataloader, model, epoch, logger, type='VAL'):
    model.model.eval()
    with torch.no_grad():
        i = 0
        all_preds = []
        all_labels = []
        for d in tqdm(dataloader,desc="{0}/EPOCH{1:02d}".format(type, epoch+1)):
            i += 1
            texts = d['texts']
            labels = d['labels']
            choices = d['choices']
            
            true, p = model.eval(texts, labels, choices)
            all_preds.extend(p.cpu().tolist())
            all_labels.extend(true.cpu().tolist())
        f1 = f1_score(all_labels, all_preds,average='weighted')
        recall = recall_score(all_labels, all_preds,average='weighted')
        print('##{0}## epoch: {1:02d}, f1: {2:.4f}, recall: {3:.4f}, precision: {4:.4f}'.format(type, epoch+1, f1, recall, precision))  
        logger.info('##{0}## epoch: {1:02d}, f1: {2:.4f}, recall: {3:.4f}, precision: {4:.4f}'.format(type, epoch+1, f1, recall, precision))
        return f1

# 固定随机种子函数
def set_seed(seed):
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True  # 使得CUDNN使用确定性算法
    torch.backends.cudnn.benchmark = False     # 如果想要可重复的结果

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--pre_train_model", type=str, default="gemma2-2b-it") 
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--weight_decay", type=float, default=0) 
    parser.add_argument("--add_bias_layer_idx", type=list, default=[5]) 
    parser.add_argument("--dataset_name", type=str, choices=["tweetHate",'sst5','bbq-age','bbq-ses','bbq-disability'], default="bbq-gender")

    parser.add_argument("--train_data_path", type=str, default="data/BBQ/data_age_train.jsonl") 
    parser.add_argument("--test_data_path", type=str, default="data/BBQ/data_age_test.jsonl")

    args = parser.parse_args()
    
    set_seed(42)
    model = finetune_Model2(args)
    mark_only_bias_as_trainable(model.model, args)
    
    optimizer = torch.optim.AdamW(model.model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    
    train_dataset = Generate_Dataset(args.train_data_path)
    test_dataset = Generate_Dataset(args.test_data_path)
    train_data = DataLoader(train_dataset, batch_size=args.batchsize, collate_fn=train_dataset.collate, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=args.batchsize, collate_fn=test_dataset.collate, shuffle=True)

    logger = get_logger('saved_logging/' + args.pre_train_model + '_' + args.dataset_name +'_'+str(args.lr)+'_'+str(args.epochs)+'_logging.log')
    logger.info(args)

    max_val_avg_acc = 0
    for epoch in range(args.epochs):
        val_avg_acc = train_eval(train_data, test_data, model, optimizer, epoch, logger) 
        
        if val_avg_acc > max_val_avg_acc:
            max_val_avg_acc = val_avg_acc
            
            #save model
            state_dict = model.model.state_dict()

            for d in state_dict.keys():
                layer = 'model.layers.{}.'.format(args.add_bias_layer_idx[0] + 1)
                if 'bias_' in d or layer+'self_attn.q_proj' in d or layer+'self_attn.v_proj' in d:
                    print(d)
                    bias = state_dict[d]
                    torch.save(bias, 'saved_models/{0}_{1}_{2}_{3}_{4}_all.pth'.format(args.pre_train_model, args.dataset_name, args.lr, str(args.add_bias_layer_idx), d))
    
    logger.info('Training is over ...')
    logger.info("The final results: {:.4f}".format(max_val_avg_acc))

    print('Training is over ...')
    print("The final results: {:.4f}".format(max_val_avg_acc))
    
