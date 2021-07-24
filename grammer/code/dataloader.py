import pickle
from tokenizer import SC_tokenizer 
from tqdm import tqdm 
import torch 
from torch.utils.data import DataLoader
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
class MyDataLoader:
    def __init__(self):
        self.tokenizer = SC_tokenizer.from_pretrained(MODEL_NAME)
        self.train = []
        self.test = []
        self._load_dataset()
#         print(self.train[0])
        
    def _load_dataset(self):
        filepath = ["./dataset/train.txt", "./dataset/test.txt"]
        for i, file in enumerate(filepath):
            with open(file, "rb") as f:
                if i == 0:
                    self.train = pickle.load(f)
                elif i == 1:
                    self.test = pickle.load(f)
    def _create_dataset_for_loader(self,
                                  dataset: list,
                                  max_length: int) -> list:
        dataloader = []
        for data in tqdm(dataset):
            wrong = data["wrong_text"]
            correct = data["correct_text"]
            # トークン化してlabelsを付与
            encoding = self.tokenizer.encode_plus_tagged(
                wrong, correct, max_length=max_length
            )
            encoding = {k: torch.tensor(v) for k, v in encoding.items()}
            dataloader.append(encoding)
        return dataloader
    def DataLoader(self, max_length: int, batch_size: int):
        """
        return: 
            32文章で32単語、1つずつのトークン
            dataloader[0]["input_ids"] = (32, 32, 1)
        """
        if len(self.train) != 0:
            encoding_train_val = self._create_dataset_for_loader(
                self.train, max_length
            )

            n = len(encoding_train_val)
            n_train = int(n*.8)
            encoding_train = encoding_train_val[:n_train]
            encoding_val = encoding_train_val[n_train:]
            # batch分だけ分割する
            dataloader_train = DataLoader(
                encoding_train, batch_size=batch_size, shuffle=True
            )
            dataloader_val = DataLoader(
                encoding_val, batch_size=256
            )
#             for i in self.test:
#                 print(i)
#                 break
            return dataloader_train, dataloader_val, self.test
        else:
            raise NotImplementedError
