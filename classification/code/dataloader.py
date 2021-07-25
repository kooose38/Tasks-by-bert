import random 
import pickle 
from torch.utils.data import DataLoader 
from tokenizer import tokenizer_ 
import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
class DataLoader_:
    def __init__(self):
        self.train, self.val, self.test = [], [], []
        self._load_dataset()
        self.tokenizer = tokenizer_()
#         print(self.test[0], len(self.test))
    def _load_dataset(self):
        filename = ["dataset/train.txt", "dataset/test.txt"]
        for i, file in enumerate(filename):
            with open(file, "rb") as f:
                text_list = pickle.load(f)
                if i == 0:
                    n = len(text_list)
                    n_train = int(n*.8)
                    self.train = text_list[:n_train]
                    self.val = text_list[n_train:]
                elif i == 1:
                    self.test = text_list 
    def _create_dataloader(self, dataset: list, max_length: int, cuda: bool) -> list:
        dataset_for_loader = []
        if len(dataset) != 0:
            for data in dataset:
                text = data["sentence"]
                label = data["labels"]
                encoding = self.tokenizer(
                    text, 
                    max_length=128,
                    padding="max_length",
                    truncation=True
                )
                encoding["labels"] = label
                encoding = {k: torch.tensor(v) for k, v in encoding.items()}
                if cuda:
                    encoding = {k: v.cuda() for k, v in encoding.items()}
                dataset_for_loader.append(encoding)
            return dataset_for_loader
        else:
            raise NotImplementedError 
    def dataloader(self, max_length: int, cuda: bool, batch_size: int):
        encoding_train = self._create_dataloader(
            self.train, max_length, cuda
        )
        encoding_val = self._create_dataloader(
            self.val, max_length, cuda
        )
        encoding_test = self._create_dataloader(
            self.test, max_length, cuda
        )
        # batch-sizeで分割する
        dataloader_train = DataLoader(
            encoding_train, batch_size=batch_size, shuffle=True
        )
        dataloader_val = DataLoader(
            encoding_val, batch_size=256
        )
        dataloader_test = DataLoader(
            encoding_test, batch_size=256
        )
#         for i in dataloader_train:
#             model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=10)
#             with torch.no_grad():
#                 output = model(**i)
#                 scores = output.logits
                
#             print(scores.argmax(-1))
            
            
#             break
        return dataloader_train, dataloader_val, dataloader_test
# data = DataLoader_()
# data.dataloader(32, False, 32)
        
        
        