import torch
import re
from transformers import BertForSequenceClassification, BertTokenizer
from config import INTENT_LABELS, PREDEFINED_PARAMS, KG_PARAM_TYPES

class IntentPredictor:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.eval()  #推理模式

    #识别文本意图，提取参数
    def parse(self, text):
        intent = self.predict(text)   #BERT预测意图

        params = {}

        if intent in PREDEFINED_PARAMS:
            param_value = self.extract_keyword(text,PREDEFINED_PARAMS[intent].keys())

            if param_value:
                params[intent] = PREDEFINED_PARAMS[intent][param_value]
        elif intent in KG_PARAM_TYPES:
            params[f"{intent}_text"] = text

        return intent, params

    def predict(self, text, confidence_threshold=0.7):

        text = re.sub(r"[^\w\s]", "", text)
        processed_text = text.lower().strip()

        inputs = self.tokenizer(   #更改text格式
            processed_text,    #文本
            return_tensors="pt",   #返回PyTorch张量
            padding=True,      #自动填充
            truncation=True,    #截断文本
            max_length=128     #最长长度
        ).to(self.device)     #数据迁移到cuda

        #模型预测
        with torch.no_grad():    #预测时关闭梯度，节省内存
            outputs = self.model(**inputs)
            pred = torch.argmax(outputs.logits).item()

        return INTENT_LABELS[pred]     #返回预测意图

    def extract_keyword(text, keywords):

        for key in keywords:
            if key in text:
                return key

        return None
