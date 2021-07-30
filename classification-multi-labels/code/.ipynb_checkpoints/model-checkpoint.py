from transformers import BertModel 
import torch 
import torch.nn as nn 

class BertForSequenceClassificationMultiLabel(nn.Module):
    def __init__(self, tag_size):
        super(BertForSequenceClassificationMultiLabel, self).__init__()
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        # bert model 
        self.bert = BertModel.from_pretrained(model_name)
        # linear layers 
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, tag_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
#         labels = x.pop("labels")
        # embedding layer 
        # transform size (batch, token, 1) -> (batch, token, 768)
        output = self.bert(**x).last_hidden_state 
        # average by batch size -> (batch, 768) 1 sequence 
        output = (output*x["attention_mask"].unsqueeze(-1)).sum(1) / x["attention_mask"].sum(1, keepdim=True)
        # linear layer -> (batch, target_size)
        output = self.fc(output)
        return output