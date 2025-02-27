import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

labels_token_tweetEmotion = [' anger',' anticipation',' disgust', ' fear', ' joy', ' love', ' optimism', ' pessimism', ' sadness', ' surprise', ' trust']
labels_token_goEmotions = [' admiration', ' amusement',' anger', ' annoyance', ' approval', ' caring', ' confusion', ' curiosity', ' desire', ' disappointment', 
                ' disapproval', ' disgust', ' embarrassment', ' excitement', ' fear', ' gratitude', ' grief', ' joy', ' love', ' nervousness', 
                ' optimism', ' pride', ' realization', ' relief', ' remorse', ' sadness', ' surprise', ' neutral']
labels_token_empDialogues = [' afraid',' angry',' annoyed',' ashamed', ' anticipating',' anxious',' apprehensive',
    ' confident',' caring', ' content', ' disappointed',' disgusted',' devastated', ' embarrassed', ' excited',
    ' faithful',' furious',' grateful',' guilty',' hopeful',' impressed',' jealous',' joyful',' lonely',' nostalgic',
    ' proud',' prepared',' sentimental',' sad',' surprised',' terrified',' trusting']
labels_token_sst2 = [' negative', ' positive']
labels_token_agnews = [' world', ' sports', ' business', ' technology']
labels_token_rte = [' True', ' False']

prompt1 = '''Review: {review} 
Emotion:'''

prompt2 = '''Utterance: {review} 
Emotion:'''

prompt3 = '''Review: {review} 
Sentiment:'''

prompt4 = '''Instruction: Select the right emotion words for the given Review from Choices.
Choices:{choice}
Review: {review}
Emotion:'''

prompt5 = '''Instruction: Select the right emotion words for the given Utterance from Choices.
Choices:{choice}
Utterance: {review}
Emotion:'''

prompt6 = '''Instruction: Select the right sentiment word for the given Review from Choices.
Choices:{choice}
Review: {review}
Sentiment:'''

prompt7 = '''Instruction: Classify the following news article from the given Choices.
Choices:{choice}
Text: {review}
Category:''' #agnews

prompt8 = '''Context: {sentence1}
Question: {sentence2} True or False?
Answer:''' #rte


prompt_generate1 = '''Review: {review} 
Emotion:{choice}'''

prompt_generate2 = '''Review: {review} 
Sentiment:{choice}'''

prompt_generate3 = '''Context: {context}
Question: {question}
Answer:{choice}'''

prompt_generate4 = '''Instruction: Select the right sentiment label for the given Review from Choices.
Choices: {option}
Review: {review}
Answer:{choice}'''

prompt_generate5 = '''For the subsequent context and question, decide on the most appropriate answer from the given options. 
Context: {context} 
Question: {question} 
Options: {option} 
Answer:{choice}'''

prompt_generate6 = '''Instruction: Select the right emotional label for the given Review from Choices.
Choices: {option}
Review: {review}
Answer:{choice}'''

class Emotion_Dataset(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        if data_path.split('/')[1] == 'tweetEmotion':
            choice = ','.join(labels_token_tweetEmotion)
            self.PROMPT = prompt4.replace('{choice}', choice)
        elif data_path.split('/')[1] == 'goEmotions':
            choice = ','.join(labels_token_goEmotions)
            self.PROMPT = prompt4.replace('{choice}', choice)
        elif data_path.split('/')[1] == 'empDialogues':
            choice = ','.join(labels_token_empDialogues)
            self.PROMPT = prompt5.replace('{choice}', choice)
        elif data_path.split('/')[1] == 'sst2':
            choice = ','.join(labels_token_sst2)
            self.PROMPT = prompt6.replace('{choice}', choice)
        elif data_path.split('/')[1] == 'agnews':
            choice = ','.join(labels_token_agnews)
            self.PROMPT = prompt7.replace('{choice}', choice)
        elif data_path.split('/')[1] == 'rte':
            self.PROMPT = prompt8
        
        if data_path.split('/')[1] == 'rte':
            self.texts, self.labels = self.read_file_rte(self.path)
        else:   
            self.texts, self.labels = self.read_file(self.path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx],
    
    def read_file(self, path):
        texts = []
        labels = []
        with open(path, 'r',encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 1 in data['gold_label_list']:
                    texts.append(self.PROMPT.format(review=data['text']))
                    labels.append(np.array(data['gold_label_list']))
        labels = np.array(labels)
        return texts, labels
    
    def read_file_rte(self, path):
        texts = []
        labels = []
        with open(path, 'r',encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 1 in data['gold_label_list']:
                    texts.append(self.PROMPT.format(sentence1=data['sentence1'], sentence2=data['sentence2']))
                    labels.append(np.array(data['gold_label_list']))
        labels = np.array(labels)
        return texts, labels
    
    def collate(self, batch):
        text,labels = zip(*batch)
       
        labels = torch.IntTensor(labels)
        return {'texts':text, 'labels':labels}

class Generate_Dataset(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        if data_path.split('/')[1] == 'tweetHate':
            self.PROMPT_GEN = prompt_generate6
        elif data_path.split('/')[1] == 'sst5':
            self.PROMPT_GEN = prompt_generate4
        else:
            self.PROMPT_GEN = prompt_generate5

        if data_path.split('/')[1] == 'BBQ':
            self.texts, self.labels, self.choices = self.read_file_bbq(self.path)
        else:
            self.texts, self.labels, self.choices = self.read_file(self.path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.choices[idx]
    
    def read_file(self, path):
        texts = []
        labels = []
        choices = []
        with open(path, 'r',encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts.append(self.PROMPT_GEN.replace('{review}', data['text']).replace('{option}',', '.join(list(data['choices'].values()))))
                labels.append(data['label'])
                choices.append(data['choices']) #确保data['choices']为字典形式
        return texts, labels, choices
    
    def read_file_bbq(self, path):
        texts = []
        labels = []
        choices = []
        with open(path, 'r',encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                options = list(data['choices'].values())
                op = ', '.join(options)
                texts.append(self.PROMPT_GEN.replace('{context}', data['context']).replace('{question}', data['question']).replace('{option}',op))
                labels.append(data['label'])
                choices.append(data['choices']) #确保data['choices']为字典形式
        return texts, labels, choices
    
    def collate(self, batch):
        text,labels,choices = zip(*batch)
       
        labels = torch.IntTensor(labels)
        return {'texts':text, 'labels':labels, 'choices':choices}

# # # 创建自定义数据集实例
# path = 'data/tweetEmotion_test.jsonl'
# my_dataset = Emotion_Dataset(path)

# # 使用DataLoader加载自定义数据集my_dataset
# dataloader = DataLoader(dataset=my_dataset, batch_size=2, collate_fn=my_dataset.collate)

# for d in dataloader:
#     #print(d[0])
#     print(d)
#     break
