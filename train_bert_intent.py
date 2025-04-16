import pandas as pd
from datasets import Dataset as HFDataset
import re
from config import INTENT_LABELS
from sklearn.model_selection import train_test_split, cross_val_score
from transformers import BertForSequenceClassification, BertTokenizer,Trainer,TrainingArguments

label2id = {label: idx for idx, label in enumerate(INTENT_LABELS)}  # 修改：添加 label2id 映射
id2label = {idx: label for idx, label in enumerate(INTENT_LABELS)}  # 修改：添加 id2label 映射

def load_dataset(data_path, tokenizer):
    df = pd.read_csv(data_path)    #读取csv文件，将csv文件转换为DataFrame格式
    df['label'] = df['label'].map(label2id)  # 将label转换为数字id

    if 'text' not in df.columns or 'label' not in df.columns:    #检查数据
        raise ValueError("CSV文件中需包含'text'和'label'列")

    dataset = HFDataset.from_pandas(df)     #将DataFrame格式转换为模型可读的Dataset格式

    def tokenize_fn(examples):    #将文本转换为BERT需要的输入格式
        return tokenizer(
            examples["text"],
            padding="max_length",  # 填充到最大长度
            truncation=True,  # 截断过长文本
            max_length=128,  # 最长长度限制
            return_tensors="pt"  # 返回PyTorch张量
        )

    dataset = dataset.map(tokenize_fn, batched=True)  #使用map方法，将所有文本进行分词编码
    return dataset   #load_dataset函数返回供模型训练的datast

def split_dataset(dataset,test_size=0.2):

    df = dataset.to_pandas()  #train_test_split函数需要DataFrame格式输入，先将原本的dataset格式转换为df
    train_df, eval_df = train_test_split(df, test_size=test_size)  #分离训练数据集与验证数据集
    train_ds = HFDataset.from_pandas(train_df)    #将DataFrame转换为dataset
    eval_ds = HFDataset.from_pandas(eval_df)

    return train_ds,eval_ds

def load_bert_model(model_path=r"C:\bert-base-chinese"):


    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(INTENT_LABELS),
        label2id=label2id,
        id2label=id2label)

    return model, tokenizer

def train_model(model, train_ds,eval_ds=None, output_dir="./intent_model"):

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,     #训练batch size
        num_train_epochs=20,          #训练轮数
        eval_strategy="steps" if eval_ds else "no",    #评估策略
        eval_steps=len(train_ds)//16,         #每epoch评估一次
        save_strategy="epoch",            #每epoch保存
        logging_steps=len(train_ds)//32,  #每epoch记录2次日志
        learning_rate=3e-5,              #学习率
        weight_decay=0.01                #权重衰减
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(output_dir)
    return trainer    #返回训练好的模型


