import torch
import torch.nn as nn
import argparse
import torch
import sys
import os
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from dataset import Emotion_Dataset
import torch.nn as nn
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score,f1_score,recall_score
torch.cuda.empty_cache()
sys.path.append(os.getcwd())

from models.model import finetune_Model

labels_token_empDialogues = [' afraid',' angry',' annoyed',' ashamed', ' anticipating',' anxious',' apprehensive',
    ' confident',' caring', ' content', ' disappointed',' disgusted',' devastated', ' embarrassed', ' excited',
    ' faithful',' furious',' grateful',' guilty',' hopeful',' impressed',' jealous',' joyful',' lonely',' nostalgic',
    ' proud',' prepared',' sentimental',' sad',' surprised',' terrified',' trusting']
labels_token_tweetEmotion = [' anger',' anticipation',' disgust', ' fear', ' joy', ' love', ' optimism', ' pessimism', ' sadness', ' surprise', ' trust']
labels_token_goEmotions = [' admiration', ' amusement',' anger', ' annoyance', ' approval', ' caring', ' confusion', ' curiosity', ' desire', ' disappointment', 
                ' disapproval', ' disgust', ' embarrassment', ' excitement', ' fear', ' gratitude', ' grief', ' joy', ' love', ' nervousness', 
                ' optimism', ' pride', ' realization', ' relief', ' remorse', ' sadness', ' surprise', ' neutral']
labels_token_sst2 = [' negative', ' positive']
labels_token_agnews = [' world', ' sports', ' business', ' technology']
labels_token_rte = [' True', ' False']
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
    
def get_pre_label(probs, labels, token):
    mask = torch.ones_like(probs, dtype=bool)  # 创建一个全为True的掩码
    mask[:,token] = False  # 把要保留的索引位置设置为False
    probs[mask] = -1000

    p = torch.zeros((probs.size(0), len(LABELS)),dtype=int)
    true_label_nums = torch.sum(labels,dim=-1)
    for j in range(probs.size(0)):
        indices = torch.argsort(-probs[j])[:int(true_label_nums[j])]
        for e in range(len(token)):
            if token[e] in indices:
                p[j,e] = 1
    return p
    
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

        loss = model.forward(texts, labels)
        
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()  
        
        loss = loss.item()
        all_loss += loss
       
        torch.cuda.empty_cache()
        logger.info('epoch: {0:02d}, batch: {1:03d}, loss: {2:.4f}'.format(epoch+1, i, loss))
    
    logger.info('##TRAIN## epoch: {0:02d}, avg_loss: {1:.4f}, avg_accuracy: {2:.4f}, avg_f1: {3:.4f}, avg_recall: {4:.4f}, avg_precision: {5:.4f}'.format(epoch+1, all_loss/i))
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

            probs, token = model.eval(texts)
          
            pre_label = get_pre_label(probs, labels, token)
            all_preds.extend(pre_label.tolist())
            all_labels.extend(labels.cpu().tolist())
            
            del probs
            torch.cuda.empty_cache()
        f1 = f1_score(all_labels, all_preds,average='weighted')
        recall = recall_score(all_labels, all_preds,average='weighted')
        precision = precision_score(all_labels, all_preds,average='weighted')
        print('##{0}_mask## epoch: {1:02d}, f1: {2:.4f}, recall: {3:.4f}, precision: {4:.4f}'.format(type, epoch+1, f1, recall, precision))
        logger.info('##{0}_mask## epoch: {1:02d}, f1: {2:.4f}, recall: {3:.4f}, precision: {4:.4f}'.format(type, epoch+1, f1, recall, precision))
        return f1

# 固定随机种子函数
def set_seed(seed):
    #random.seed(seed)
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
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15) 
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--add_bias_layer_idx", type=list, default=[14])
    
    parser.add_argument("--dataset_name", type=str, choices=["tweetEmotion", "goEmotions", "empDialogues", "sst2", "agnews", "rte"], default="tweetEmotion")                                                                                                                                                                                                                      # parser.add_argument("--dataset_name", type=str, choices=["tweetEmotion", "goEmotions"], default="tweetEmotion")
    parser.add_argument("--train_data_path", type=str, default="data/tweetEmotion/data_tweet_emotion_train.jsonl") #tweet, go 
    parser.add_argument("--test_data_path", type=str, default="data/tweetEmotion/data_tweet_emotion_test.jsonl")
    
    args = parser.parse_args()
    
    set_seed(42)

    if args.dataset_name == 'tweetEmotion':
        LABELS = labels_token_tweetEmotion
    elif args.dataset_name == 'goEmotions':
        LABELS = labels_token_goEmotions
    elif args.dataset_name == 'empDialogues':
        LABELS = labels_token_empDialogues
    elif args.dataset_name == 'sst2':
        LABELS = labels_token_sst2
    elif args.dataset_name == 'agnews':
        LABELS = labels_token_agnews
    elif args.dataset_name == 'rte':
        LABELS = labels_token_rte
    else:
        raise Exception('没有匹配的TOKEN参数！')
    
    model = finetune_Model(args, LABELS)

    mark_only_bias_as_trainable(model.model, args)
     
    optimizer = torch.optim.AdamW(model.model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    
    train_dataset = Emotion_Dataset(args.train_data_path)
    test_dataset = Emotion_Dataset(args.test_data_path)
    train_data = DataLoader(train_dataset, batch_size=args.batchsize, collate_fn=train_dataset.collate, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=args.batchsize, collate_fn=test_dataset.collate, shuffle=True)

    logger = get_logger('saved_logging/' + args.pre_train_model + '_' + args.dataset_name +'_'+str(args.lr)+'_'+str(args.epochs)+'_logging.log')
    logger.info(args)

    max_val_avg_f1 = 0
    for epoch in range(args.epochs):
        val_avg_f1 = train_eval(train_data, test_data, model, optimizer, epoch, logger) 
        
        if val_avg_f1 > max_val_avg_f1:
            max_val_avg_f1 = val_avg_f1
            print(max_val_avg_f1)
            
            #save model
            state_dict = model.model.state_dict()

            for d in state_dict.keys():
                layer = 'model.layers.{}.'.format(args.add_bias_layer_idx[0] + 1)
                if 'bias_' in d or layer+'self_attn.o_proj' in d or layer+'mlp.down_proj' in d:
                    print(d)
                    bias = state_dict[d]
                    torch.save(bias, 'saved_models/{0}_{1}_{2}_{3}_{4}4.pth'.format(args.pre_train_model, args.dataset_name, args.lr, str(args.add_bias_layer_idx), d))
               
    logger.info('Training is over ...')
    logger.info("The final results: {:.4f}".format(max_val_avg_f1))

    print('Training is over ...')
    print("The final results: {:.4f}".format(max_val_avg_f1))
    
