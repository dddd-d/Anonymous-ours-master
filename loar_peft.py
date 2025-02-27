import random
import transformers
from trl import SFTTrainer
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
os.environ["WANDB_DISABLED"] = "true"
MODEL = 'gemma2-2b-it'
DATASET = 'tweetHate'

prompt1 = '''Review: {review}
Emotion:{choice}''' #tweetEmotion,goEmotions
prompt2 = '''Utterance: {review}
Emotion:{choice}''' #empDialogues
prompt3 = '''Review: {review}
Sentiment:{choice}''' #sst2

prompt_bbq = '''Context: {context}
Question: {question}
Answer: {choice}'''
prompt4 = '''Review: {context}
Sentiment: {choice}''' #sst5
prompt5 = '''Review: {context}
Emotion: {choice}''' #tweetHate

prompt_instr1 = '''Instruction: Select the right emotion words for the given Review from Choices.
Choices:{choices}
Review: {review}
Emotion:{choice}''' #tweetEmotion, goEmotions

prompt_instr2 = '''Instruction: Select the right emotion words for the given Utterance from Choices.
Choices:{choices}
Utterance: {review}
Emotion:{choice}''' #empDialogues

prompt_instr3 = '''Instruction: Select the right sentiment word for the given Review from Choices.
Choices:{choices}
Review: {review}
Emotion:{choice}''' #sst2

prompt_instr4 = '''Instruction: Select the right sentiment label for the given Review from Choices.
Choices: {option} 
Review: {review} 
Answer: {choice}''' #sst5

prompt_instr5 = '''Instruction: Select the right emotional label for the given Review from Choices.
Choices: {option} 
Review: {review} 
Answer: {choice}''' #hate

prompt_instr6 = '''Context: {sentence1}
Question: {sentence2} True or False?
Answer:{choice}''' #rte

prompt_instr7 = '''Instruction: Classify the following news article from the given Choices.
Choices:{choices}
Text: {review}
Category:{choice}''' #agnews

prompt_instr8 = '''For the subsequent context and question, decide on the most appropriate answer from the choices available. 
Context: {context} 
Question: {question} 
Options: {option} 
Answer: {choice}''' #bbq

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

if DATASET == 'goEmotions':
    TOKEN = labels_token_goEmotions
    #PROMPT = prompt1
    PROMPT = prompt_instr1.replace('{choices}', ','.join(TOKEN))
    print(PROMPT)
    MAX_LENGTH = 110
    train_path = 'data/goEmotions/data_go_emotions_validation_demo.jsonl' #1000(imbalance)
    test_path = 'data/goEmotions/data_go_emotions_test.jsonl' # 5427
elif DATASET == 'tweetEmotion':
    TOKEN = labels_token_tweetEmotion
    PROMPT = prompt_instr1.replace('{choices}', ','.join(TOKEN))
    #PROMPT = prompt1
    print(PROMPT)
    MAX_LENGTH = 80
    train_path = 'data/tweetEmotion/data_tweet_emotion_train2.jsonl' #886 (imbalance)
    test_path = 'data/tweetEmotion/data_tweet_emotion_test2.jsonl' # 3259
elif DATASET == 'empDialogues':
    TOKEN = labels_token_empDialogues
    PROMPT = prompt_instr2.replace('{choices}', ','.join(TOKEN))
    #PROMPT = prompt2
    MAX_LENGTH = 125
    train_path = 'data/empatheticdialogues/data_emp_dialogues_train.jsonl' #960 (balance)
    test_path = 'data/empatheticdialogues/data_emp_dialogues_test.jsonl' #2538
elif DATASET == 'sst2':
    TOKEN = labels_token_sst2
    #PROMPT = prompt3
    PROMPT = prompt_instr3.replace('{choices}', ','.join(TOKEN))
    print(PROMPT)
    MAX_LENGTH = 64
    train_path = 'data/sst2/data_sst2_validation.jsonl' # 872 (balance)
    test_path = 'data/sst2/data_sst2_test.jsonl' #1821
elif DATASET == 'tweetHate':
    #PROMPT = prompt5
    PROMPT = prompt_instr5
    MAX_LENGTH = 80
    train_path = 'data/tweetHate/data_tweet_hate_train.jsonl' #522
    test_path = 'data/tweetHate/data_tweet_hate_test.jsonl' #1433
elif DATASET == 'sst5':
    #PROMPT = prompt4
    PROMPT = prompt_instr4
    MAX_LENGTH = 80
    train_path = 'data/sst5/data_sst5_validation.jsonl' #1101 
    test_path = 'data/sst5/data_sst5_test.jsonl' #2210
elif DATASET.split('-')[0] == 'bbq':
    PROMPT = prompt_instr8
    MAX_LENGTH = 120
    train_path = 'data/BBQ/data_{}_train.jsonl'.format(DATASET.split('-')[1]) #10% 368 686 155
    test_path = 'data/BBQ/data_{}_test.jsonl'.format(DATASET.split('-')[1])
elif DATASET == 'rte':
    TOKEN = labels_token_rte
    PROMPT = prompt_instr6
    MAX_LENGTH = 95
    train_path = 'data/rte/data_rte_train.jsonl'
    test_path = 'data/rte/data_rte_test.jsonl'
elif DATASET == 'agnews':
    TOKEN = labels_token_agnews
    PROMPT = prompt_instr7.replace('{choices}', ','.join(TOKEN))
    MAX_LENGTH = 90
    train_path = 'data/agnews/data_agnews_train.jsonl'
    test_path = 'data/agnews/data_agnews_test.jsonl'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map='auto')
#model = AutoModelForCausalLM.from_pretrained(MODEL,device_map = 'auto')

from datasets import load_dataset

def map_gold_labels(example):
    labels = [TOKEN[i] for i, val in enumerate(example["gold_label_list"]) if val == 1]
    if len(labels) > 1:
        idx = random.randint(0, len(labels) - 1)
        example["label_names"] = labels[idx]
    elif len(labels) == 1:
        example["label_names"] = labels[0]
    else:
        example["label_names"] = ''
    return example

def tokenize_function(examples): #sst2, goEmotion, tweetEmotion, empDialogues
    texts = [
        PROMPT.replace('{review}', text).replace('{choice}',label)
        for text, label in zip(examples['text'], examples['label_names'])
    ]
    print(texts[0])
    if MODEL in ['mistral-7b-instruct','llama3.2-1b', 'llama3.2-3b']:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",  # Ensure consistent padding
        max_length=MAX_LENGTH,  # Adjust max_length as needed
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs
 
def tokenize_function2(examples): #sst5, tweetHate
    choices = examples['choices'][0]
    texts = [
        PROMPT.replace('{review}', text).replace('{choice}',choices[str(label)]).replace('{option}', ', '.join(list(choices.values())))
        for text, label in zip(examples['text'], examples['label'])
    ]
    print(texts[0])
    if MODEL in ['mistral-7b-instruct','llama3.2-1b', 'llama3.2-3b']:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",  # Ensure consistent padding
        max_length=MAX_LENGTH,  # Adjust max_length as needed
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

def tokenize_function3(examples): #bbq
    texts = [
        PROMPT.replace('{context}', t).replace('{question}', q).replace('{choice}',c[str(l)]).replace('{option}', ', '.join(list(c.values())))
        for t, q, l, c in zip(examples['context'], examples['question'], examples['label'], examples['choices'])
    ]
    print(texts[0])
    if MODEL in ['mistral-7b-instruct','llama3.2-1b', 'llama3.2-3b']:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",  # Ensure consistent padding
        max_length=MAX_LENGTH,  # Adjust max_length as needed
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs
def tokenize_function4(examples): #rte
    texts = [
        PROMPT.replace('{sentence1}', sentence1).replace('{sentence2}', sentence2).replace('{choice}',label)
        for sentence1, sentence2, label in zip(examples['sentence1'], examples['sentence2'], examples['label_names'])
    ]
    print(texts[0])
    if MODEL in ['mistral-7b-instruct','llama3.2-1b', 'llama3.2-3b']:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",  # Ensure consistent padding
        max_length=MAX_LENGTH,  # Adjust max_length as needed
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

data = load_dataset("json", data_files=train_path)
data_test = load_dataset("json", data_files=test_path)
if DATASET in ['empDialogues','sst2','tweetEmotion','goEmotions','agnews']:
    data = data.map(map_gold_labels, load_from_cache_file=False)
    data_test = data_test.map(map_gold_labels, load_from_cache_file=False)
    data = data.map(tokenize_function, batched=True, remove_columns=["gold_label_list",'label_names'], load_from_cache_file=False)
    data_test = data_test.map(tokenize_function, batched=True, remove_columns=["gold_label_list",'label_names'], load_from_cache_file=False)
if DATASET in ['rte']:
    data = data.map(map_gold_labels, load_from_cache_file=False)
    data_test = data_test.map(map_gold_labels, load_from_cache_file=False)
    data = data.map(tokenize_function4, batched=True, remove_columns=["gold_label_list",'label_names'], load_from_cache_file=False)
    data_test = data_test.map(tokenize_function4, batched=True, remove_columns=["gold_label_list",'label_names'], load_from_cache_file=False)
elif DATASET in ['tweetHate','sst5']:
    data = data.map(tokenize_function2, batched=True, remove_columns=["choices","label"], load_from_cache_file=False)
    data_test = data_test.map(tokenize_function2, batched=True, remove_columns=["choices",'label'], load_from_cache_file=False)
elif DATASET.split('-')[0] == 'bbq':
    data = data.map(tokenize_function3, batched=True, remove_columns=["choices","label"], load_from_cache_file=False)
    data_test = data_test.map(tokenize_function3, batched=True, remove_columns=["choices",'label'], load_from_cache_file=False)

df = pd.DataFrame(data['train'])
print(df.head())
print(df.iloc[0]['input_ids'])

def formatting_func(example):
    #text = f"Review: {example['text'][0]}\nEmotion:{example['label_names'][0]}"
    #text = PROMPT.replace('{review}',example['text'][0]).replace('{choice}',example['label_names'][0])
    #print(text)
    ##PROMPT.replace('{review}', text).replace('{choice}', label)
    formatted_strings = [
        f"Review: {text}\nEmotion:{label}"
        for text, label in zip(example['text'], example['label_names'])
    ]
    print(formatted_strings)
    return formatted_strings

# 定义LoRA微调参数
from peft import LoraConfig

lora_config = LoraConfig(
    r=8, # LoRA中的秩
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    #target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",  # 因果语言模型
)

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    #eval_dataset=data_test['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,#batch size = per_device_train_batch_size * gradient_accumulation_steps * num_devices
        #per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=100,
        #num_train_epochs = 10,
        learning_rate=1e-3, #[1e-4, 5e-4]
        #weight_decay= 5e-3,
        fp16=True,
        logging_steps=1,
        output_dir="outputs/{0}_{1}".format(MODEL, DATASET),
        optim="paged_adamw_8bit",
        #eval_strategy="epoch",  # 每个epoch评估一次
        #save_strategy="epoch",  # 每个epoch保存一次
        # loa_best_model_at_end=True,  # 训练结束时加载最佳模型
    ),
    peft_config=lora_config,
    #max_seq_length=128 
    #formatting_func=formatting_func,
)
trainer.train()


# trainer = Trainer(
#     model=model,
#     train_dataset=data["train"],
#     #eval_dataset=data_test['train'],
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=8,#batch size = per_device_train_batch_size * gradient_accumulation_steps * num_devices
#         #per_device_eval_batch_size=8,
#         gradient_accumulation_steps=4,
#         warmup_steps=2,
#         max_steps=100,
#         #num_train_epochs = 10,
#         learning_rate=1e-5, #[1e-4, 5e-4]  #2b: 5e-5
#         #weight_decay= 5e-3,
#         fp16=True,
#         logging_steps=1,
#         output_dir="outputs/full_{0}_{1}".format(MODEL, DATASET),
#         optim="paged_adamw_8bit",
#         #eval_strategy="epoch",  # 每个epoch评估一次
#         #save_strategy="epoch",  # 每个epoch保存一次
#         # loa_best_model_at_end=True,  # 训练结束时加载最佳模型
#     ),
#     tokenizer = tokenizer
# )
# trainer.train()