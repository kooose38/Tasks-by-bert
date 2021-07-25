from transformers import BertForSequenceClassification
import torch
from dataloader import DataLoader_
import torch.nn as nn 

class BertForSequenceClassification_(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        """
        (文章数, 単語数, 1) 入力層
          -> (文章数, 単語数, 768) BertModel
            -> (文章数, 768) tokenから[CLS]のベクトルのみを抽出
              -> (文章数, num_labels) Linear/ ReLU/ Softmax
        """
        super(Deve, self).__init__()
        self.bert_sc = BertForSequenceClassification.from_pretrained(model_name,
                                                                    num_labels=num_labels)
    def forward(self, x):
        output = self.bert_sc(**x)
        return output.logits