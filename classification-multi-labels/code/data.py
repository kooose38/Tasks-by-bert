import torch 
from torch.utils.data import DataLoader 
from tokenizer import tokenizer_
import random 
class DataLoader_:
    def __init__(self):
        self.tokenizer = tokenizer_()
        self.dataset_train = []
        self.dataset_val = []
        self.dataset_test = []
        self._load_dataset()
    def _load_dataset(self):
        # loading dataset from ./dataset
        import pickle 
        f = open("./dataset/dataset.txt", "rb")
        data = pickle.load(f)
        n = len(data)
        n_train = int(n*.6)
        n_val = int(n*.2)
        data = random.shuffle(data)
        self.dataset_train = data[:n_train]
        self.dataset_val = data[n_train:n_train+n_val]
        self.dataset_test = data[n_train+n_val:]
    def dataloader(self, max_length: int, batch_size: int, cuda: bool):
        if len(self.dataset_train) != 0:
            # tokenizer for bert model 
            train2tensor = self._sentence2tensor(self.dataset_train, cuda, max_length)
            val2tensor = self._sentence2tensor(self.dataset_val, cuda, max_length)
            test2tensor = self._sentence2tensor(self.dataset_test, cuda, max_length)
            # seprate batch 
            train = DataLoader(train2tensor,
                               batch_size=batch_size,
                              shuffle=True) # i["input_ids"].size() for i in train == (32, 128)
            val = DataLoader(val2tensor, batch_size=256)
            test = DataLoader(test2tensor, batch_size=256)
            return train, val, test 
        else: 
            raise NotImplementedError
            
    def _sentence2tensor(self, dataset: list, cuda: bool, max_length: int):
        sentence2tensor = []
        for data in dataset:
            text = data["text"]
            labels = data["labels"]
            encoding = self.tokenizer(text,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True)
            encoding["labels"] = labels 
            encoding = {k: torch.tensor(v) for k, v in encoding.items()}
            if cuda:
                encoding = {k: v.cuda() for k, v in encoding.items()}
            sentence2tensor.append(encoding)
            
        return sentence2tensor
        