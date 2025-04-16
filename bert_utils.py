import re
from datasets import Dataset as HFDataset
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer,Trainer,TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from config import INTENT_LABELS

label2id = {label: idx for idx, label in enumerate(INTENT_LABELS)}  # 修改：添加 label2id 映射
id2label = {idx: label for idx, label in enumerate(INTENT_LABELS)}  # 修改：添加 id2label 映射

#模型加载
def load_bert_model(model_path=r"C:\bert-base-chinese"):


    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(INTENT_LABELS),
        label2id=label2id,
        id2label=id2label)

    return model, tokenizer

#数据预处理
def preprocess_text(text):


    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower().strip()

    return text



#创建数据集
def create_intent_dataset(data_path, tokenizer):
    df = pd.read_csv(data_path)
    df['label'] = df['label'].map(label2id)  #将label转换为数字id

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV文件中需包含'text'和'label'列")
    dataset = HFDataset.from_pandas(df)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",      #填充到最大长度
            truncation=True,           #截断过长文本
            max_length=128,            #最长长度限制
            return_tensors="pt"        #返回PyTorch张量
        )
    dataset = dataset.map(tokenize_fn,batched=True)
    return dataset

#模型训练
def train_intent_classifier(model,train_dataset,eval_dataset=None,output_dir="./intent_model",epochs=30):
    """

    :param model: 训练的模型
    :param train_dataset: 训练数据集
    :param eval_dataset: 验证数据集
    :param output_dir: 模型保存路径
    :param epochs: 训练轮数
    :return: 训练好的模型
    """

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,     #训练batch size
        num_train_epochs=epochs,          #训练轮数
        eval_strategy="steps" if eval_dataset else "no",    #评估策略
        eval_steps=100 if eval_dataset else None,         #每100步评估一次
        save_steps=100,                   #每100步保存一次
        logging_steps=50,                #每50步记录一次日志
        learning_rate=3e-5,              #学习率
        weight_decay=0.01                #权重衰减
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    return model

#意图预测
def predict_intent(text,model,tokenizer,device='cuda',confidence_threshold=0.7):
    """

    :param text: 输入的文本
    :param model: BertForSequenceClassification
    :param tokenizer: BertTokenizer
    :param confidence_threshold: float-置信度阈值
    :return: (str,float)-(预测的意图，置信度得分)
    """
    processed_text = preprocess_text(text)
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)

    #模型预测
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, pred = torch.max(probs, dim=1)

    #应用置信度阈值
    intent = INTENT_LABELS[pred.item()]
    confidence_value = confidence.item()

    if confidence_value < confidence_threshold:
        intent = "unknown"

    return intent, confidence_value

def test():
    model, tokenizer = load_bert_model()

    dataset = create_intent_dataset("intent_data.csv", tokenizer)
    df = dataset.to_pandas()
    train_df, eval_df = train_test_split(df,test_size=0.2)
    train_dataset = HFDataset.from_pandas(train_df)
    eval_dataset = HFDataset.from_pandas(eval_df)

    train_model = train_intent_classifier(model,train_dataset,eval_dataset)

    text = "推荐一些川菜"
    intent, confidence = predict_intent(text, train_model, tokenizer)
    print(f"意图：{intent},置信度：{confidence:.2f}")

if __name__ == "__main__":
    test()
