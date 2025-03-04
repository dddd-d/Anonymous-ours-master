import math
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

class finetune_Model2:
    def __init__(self, config):
        
        self.config = config
        self.pre_train_model = config.pre_train_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.pre_train_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.pre_train_model,device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='eager')
        self.loss_fn = nn.CrossEntropyLoss()
        self.model_config = self.model.config
        self.add_bias_layer_idx = config.add_bias_layer_idx
        self.bias_list = self.add_bias(self.add_bias_layer_idx)
        self.modify(self.add_bias_layer_idx, self.bias_list)

    def forward(self, texts, labels, choices):
        #获取一个batch的所有labels和计数
        labels_count = {}
        labels_idx = {}
        new_labels = []
        idx = 0
        for b in range(len(texts)):
            name = choices[b][str(labels[b].item())]
            if name not in labels_count.keys():
                labels_count[name] = 1
                labels_idx[name] = idx
                idx += 1
            else:
                labels_count[name] += 1

            new_labels.append(labels_idx[name])

        w = torch.tensor(list(labels_count.values()), dtype=float)  
        w = F.softmax(w, dim=-1).to(self.config.device)
        labels_list = list(labels_count.keys())
        
        probs, loss = self.get_probs(texts, labels_list, new_labels)

        extract_in = torch.zeros(len(labels_list)).to(self.config.device)
        extract_out = torch.zeros(len(labels_list)).to(self.config.device)
        for idx, name in enumerate(labels_list):
            tmp_in = 0
            tmp_out = 0
            num_in = 0
            num_out = 0
            for t in range(probs.size(0)):
                if new_labels[t] == idx:
                    tmp_in += probs[t, idx]
                    num_in += 1
                else:
                    tmp_out += probs[t, idx]
                    num_out += 1
            if num_in > 0:
                extract_in[idx] = tmp_in/num_in

            if num_out > 0:
                extract_out[idx] = tmp_out/num_out
        
        center_in = torch.mean(extract_in, dim=-1)
        center_out = torch.mean(extract_out, dim=-1)
        
        dis1 = torch.norm(extract_in - center_in)
        dis2 = torch.norm(extract_out - center_out)
        dis3 = torch.norm(extract_in - extract_out)
        
        dis = dis1 + dis2 - dis3
        L = dis + loss 
        
        del extract_out, extract_in
        torch.cuda.empty_cache()
        
        return L

    def forward_all(self, texts, labels, choices):
        loss = 0
        for k in range(len(texts)):
            text = texts[k].replace('{choice}', ' '+choices[k][str(labels[k].item())])
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.config.device)
            outputs = self.model(input_ids = input_ids, labels = input_ids) #labels = input_ids
            loss += outputs.loss
        torch.cuda.empty_cache()      
        
        return loss
        
    def batch_macro_metric(self, true_labels, pred_labels, num_classes):
        """
        计算 batch 的宏观 F1 分数，忽略在真实标签和预测标签中均未出现的类别。
        """
        # 用于存储活跃类别的 F1 值
        f1_scores = []
        recall_scores = []
        precision_scores = []
        
        # 逐类别计算 F1 分数
        for i in range(num_classes):
            true_class = true_labels[:, i].cpu().numpy()
            pred_class = pred_labels[:, i].cpu().numpy()
            
            # 仅当该类别在真实中出现时计算 F1
            if true_class.sum() > 0:
                f1 = f1_score(true_class, pred_class)
                recall = recall_score(true_class, pred_class)
                precision = precision_score(true_class, pred_class)
                f1_scores.append(f1)
                recall_scores.append(recall)
                precision_scores.append(precision)
        
        # 将活跃类别的 F1 分数取平均，计算宏观 F1
        if f1_scores:
            macro_f1 = sum(f1_scores) / len(f1_scores)
            macro_recall = sum(recall_scores) / len(recall_scores)
            macro_precision = sum(precision_scores) / len(precision_scores)
        else:
            macro_f1 = 0.0  # 如果没有活跃类别，默认为 0
            macro_recall = 0.0 
            macro_precision = 0.0

        return macro_f1, macro_recall, macro_precision

    def eval(self, texts, labels, choices): #choice dict
        self.model.eval()
        with torch.no_grad():
            p = torch.zeros((len(texts), len(choices[0])))
            true = torch.zeros((len(texts), len(choices[0])))
            true[range(true.size(0)), labels] = 1
            for k in range(len(texts)):
                PROMPT = texts[k]
                prefix = PROMPT.replace('{choice}', '')
                prefix_input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids
                score = []
                for idx, lab in choices[k].items():
                    text = PROMPT.replace('{choice}', ' '+lab)
                    input_ids = self.tokenizer(text, return_tensors="pt").input_ids
                    continue_ids = input_ids[0, prefix_input_ids.shape[-1]:]
                    outputs = self.model(input_ids = input_ids.to(self.config.device), use_cache=False)
                
                    # skip tokens in the prompt -- we only care about the answer
                    logits = outputs.logits[0, prefix_input_ids.shape[-1] - 1: -1, :]
                    logits = F.log_softmax(logits, dim=-1)
                    # get probs for each token in the answer
                    score.append(logits[range(logits.shape[0]), continue_ids].sum().item())

                max_index = score.index(max(score))
                p[k, max_index] = 1
            
            f1 = f1_score(true.cpu(), p.cpu(),average='weighted')
            recall = recall_score(true.cpu(), p.cpu(),average='weighted')
            precision = precision_score(true.cpu(), p.cpu(),average='weighted')
            
            accuracy = ((true == p).all(dim=1).sum().item())/p.size(0)
            #f1, recall, precision = self.batch_macro_metric(true, p, len(choices[0]))
        return accuracy, f1, recall, precision

    def get_probs(self, texts, labels_list, new_labels):

        probs = torch.zeros(len(texts), len(labels_list)).to(self.config.device)
        loss = 0
        for k in range(len(texts)):
            PROMPT = texts[k]
            prefix = PROMPT.replace('{choice}', '')
            prefix_input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids
            for idx, lab in enumerate(labels_list):
                text = PROMPT.replace('{choice}', ' '+lab)
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.config.device)
                continue_ids = input_ids[0, prefix_input_ids.shape[-1]:].cpu()
                outputs = self.model(input_ids = input_ids, labels = input_ids) #labels = input_ids
                if idx == new_labels[k]:
                    loss += outputs.loss
                # skip tokens in the prompt -- we only care about the answer
                logits = outputs.logits[0, prefix_input_ids.shape[-1]-1 : -1, :]
                logits = F.log_softmax(logits, dim=-1)
                # get probs for each token in the answer
                probs[k, idx] = logits[range(logits.shape[0]), continue_ids].sum()
        torch.cuda.empty_cache()        
        return probs, loss

    def add_bias(self, bias_idx):
        bias_list = []
        for i in bias_idx:
            bias = torch.zeros(self.model_config.hidden_size,dtype=torch.bfloat16)
            bias = self.reset_parameters(bias)
            bias = nn.Parameter(bias.to(self.config.device))
            self.model.register_parameter("bias_{}".format(i), bias)
            bias_list.append(bias)
        return bias_list
    
    def reset_parameters(self, bias):
        stdv = 1. / math.sqrt(self.model_config.hidden_size)  #平方根
        bias.uniform_(-stdv, stdv)
        return bias
    
    def modify(self, bias_idx, bias_list):
        for i in range(len(bias_idx)):
            assert len(bias_idx) == len(bias_list)
            
            # 保存模型原始的前向传播方法
            target_layer = self.model.model.layers[bias_idx[i]]
            original_forward = target_layer.forward
            registered_bias = getattr(self.model, 'bias_'+str(bias_idx[i]))
                
            def modified_forward(*args,  **kwargs):
                # 使用原始前向传播计算输出
                output = original_forward(*args,  **kwargs)
                # 向输出添加新的偏置项
                output = list(output)
                output[0] = output[0] + output[0] * registered_bias.to(output[0].device)

                return tuple(output)

            # 将解码层的前向传播替换为修改后的版本
            target_layer.forward = modified_forward

class finetune_Model:
    def __init__(self, config, labels_token):
        #super(myModel, self).__init__()
        
        self.config = config
        
        self.pre_train_model = config.pre_train_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.pre_train_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.pre_train_model,device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='eager')
        self.model_config = self.model.config
        self.lm_head = self.model.lm_head
        self.norm = self.model.model.norm
        self.add_bias_layer_idx = config.add_bias_layer_idx
        self.bias_list = self.add_bias(self.add_bias_layer_idx)
        self.modify(self.add_bias_layer_idx, self.bias_list)

        self.token, self.token1 = self.tokenizer_emotion(labels_token)
        self.labels_token = labels_token

    def forward_all(self, texts, labels):
        loss = 0
        for k in range(len(texts)):
            a = (labels[k] == 1).nonzero(as_tuple=True)[0]
            if a.size(-1) > 1:
                idx = random.randint(0, a.size(-1) - 1)
            else:
                idx = 0
            
            index = a[idx].item() 
            text = texts[k] + self.labels_token[index] 
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.config.device)
            outputs = self.model(input_ids = input_ids, labels = input_ids) #labels = input_ids
            loss += outputs.loss
        torch.cuda.empty_cache()      
        
        return loss

    def forward(self, texts, labels):
        loss = 0
        probs_labels = torch.zeros(len(texts), len(self.token)).to(self.config.device)
        
        for k in range(len(texts)):
            a = (labels[k] == 1).nonzero(as_tuple=True)[0]
            if a.size(-1) > 1:
                idx = random.randint(0, a.size(-1) - 1)
            else:
                idx = 0
            
            index = a[idx].item() 
            text = texts[k] + self.labels_token[index] 
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.config.device)
            prefix_input_ids = self.tokenizer(texts[k], return_tensors="pt").input_ids
            ids = prefix_input_ids.size(-1) - 1
            outputs = self.model(input_ids = input_ids, labels = input_ids)

            loss += outputs.loss

            logits = outputs.logits[0,ids,:]
            #logits = F.log_softmax(logits, dim=-1)

            for j in range(len(self.token)):
                probs_labels[k, j] = logits[self.token1[j]]

        extract_in = torch.zeros(len(self.token)).to(self.config.device)
        extract_out = torch.zeros(len(self.token)).to(self.config.device)
        for k in range(len(self.token)):
            tmp_in = 0
            tmp_out = 0
            num_in = 0
            num_out = 0
            for t in range(len(texts)):
                if labels[t, k] == 1:
                    tmp_in += probs_labels[t, k]
                    num_in += 1
                else:
                    tmp_out += probs_labels[t, k]
                    num_out += 1
            if num_in > 0:
                extract_in[k] = tmp_in/num_in

            if num_out > 0:
                extract_out[k] = tmp_out/num_out

        dis1 = 0
        dis2 = 0
        dis3 = 0
        center_in = torch.mean(extract_in[extract_in != 0], dim=-1)
        center_out = torch.mean(extract_out[extract_out != 0], dim=-1)
        for i in range(extract_in.size(-1)):
            if extract_in[i] != 0:
                dis1 += (extract_in[i] - center_in)**2
            if extract_out[i] != 0:
                dis2 += (extract_out[i] - center_out)**2
            if extract_in[i] != 0 and extract_out[i] != 0:
                dis3 += (extract_in[i] - extract_out[i])**2

        dis1 = dis1 ** 0.5
        dis2 = dis2 ** 0.5
        dis3 = dis3 ** 0.5

        dis = dis1 + dis2 - dis3
         
        del input_ids, extract_out, extract_in, probs_labels
        torch.cuda.empty_cache()
        print(loss.item(), dis.item())
        return dis + loss
    
    def eval(self, texts):
        self.model.eval()
        with torch.no_grad():
            input_ids = self.tokenizer_text(texts)
            probs = torch.zeros(len(input_ids), self.model_config.vocab_size).to(self.config.device)
            for i in range(len(input_ids)):
                outputs = self.model(input_ids = input_ids[i], use_cache=False)
                logits = outputs.logits[0,-1,:]
                probs[i,:] = logits
        return probs, self.token1
    
    def tokenizer_text(self, texts):
        token = []
        for k in texts:
            input_ids = self.tokenizer(k, return_tensors="pt").input_ids.to(self.config.device)
            token.append(input_ids)
        return token
    
    def tokenizer_emotion(self, labels):
        token = []
        token1 = []
        for k in labels:
            emotion_ids1 = self.tokenizer(k, return_tensors="pt").input_ids[0,1]
            emotion_ids = self.tokenizer(k, return_tensors="pt").input_ids[0,1:]
            token.append(emotion_ids)
            token1.append(emotion_ids1)
        return token, token1

    def add_bias(self, bias_idx):
        bias_list = []
        bias = torch.zeros(self.model_config.hidden_size,dtype=torch.bfloat16)
        bias = self.reset_parameters(bias)
        bias = nn.Parameter(bias.to(self.config.device))
        self.model.register_parameter("bias_{}".format(bias_idx[0]), bias)
        bias_list.append(bias)
        return bias_list
    
    def reset_parameters(self, bias):
        stdv = 1. / math.sqrt(self.model_config.hidden_size)  #平方根
        bias.uniform_(-stdv, stdv)
        return bias
    
    def modify(self, bias_idx, bias_list):
            
        # 保存模型原始的前向传播方法
        target_layer = self.model.model.layers[bias_idx[0]]
        original_forward = target_layer.forward
        registered_bias = getattr(self.model, 'bias_'+str(bias_idx[0]))
        #registered_bias = getattr(self.model, 'weight_'+str(bias_idx[i]))
            
        def modified_forward(*args,  **kwargs):
            # 使用原始前向传播计算输出
            output = original_forward(*args,  **kwargs)
            # 向输出添加新的偏置项
            output = list(output)
            output[0] = output[0] + output[0] * registered_bias.to(output[0].device)
            return tuple(output)

        # 将解码层的前向传播替换为修改后的版本
        target_layer.forward = modified_forward

        
