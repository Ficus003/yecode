import torch
import re
from transformers import BertForSequenceClassification, BertTokenizer
from config import INTENT_LABELS

class IntentPredictor:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def predict(self, text, confidence_threshold=0.7):

        text = re.sub(r"[^\w\s]", "", text)
        processed_text = text.lower().strip()

        inputs = self.tokenizer(   #更改text格式
            processed_text,    #文本
            return_tensors="pt",   #返回PyTorch张量
            padding=True,      #自动填充
            truncation=True,    #截断文本
            max_length=128     #最长长度
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self.model.to(self.device)     #数据迁移到cuda

        #模型预测
        with torch.inference_mode():    #预测时关闭梯度，节省内存
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)  #outputs.logits模型输出的原始得分,softmax将得分转换成概率分布
            confidence, pred = torch.max(probs, dim=1)     #寻找概率最大的类别，返回预测标签与置信度

            # 应用置信度阈值
        intent = INTENT_LABELS[pred.item()]
        confidence_value = confidence.item()

        if confidence_value < confidence_threshold:     #如果置信度小于置信度阈值0.7则返回"unknown",防止系统乱猜
            intent = "unknown"

        return {
            "intent":intent,
            "confidence":confidence_value}     #返回预测意图与置信度,flask接口需要字典格式


