from model import BertForMaskedLM_pl
import pytorch_lightning as pl
from dataloader import MyDataLoader 
import sys 
from tqdm import tqdm 
import predict 

args = sys.argv
max_length = args[1]
max_length = int(max_length)
batch_size = args[2]
batch_size = int(batch_size)
lr = args[3]
lr = float(lr)

class Trainer:
    def __init__(self):
        self.dataloader_train,  self.dataloader_val, self.dataset_test = [], [], []
    def load_dataloader(self, max_length=32, batch_size=32):
        data = MyDataLoader()
        self.dataloader_train, self.dataloader_val, self.dataset_test = data.DataLoader(max_length, batch_size)
#         print("Dataset loading is complete!!!")
    def _trainer(self, cuda: bool):
        
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_weights_only=True,
            dirpath='model/'
        )

        trainer = pl.Trainer(
            gpus=1 if cuda else 0,
            max_epochs=3,
            callbacks=[checkpoint]
        )
        return trainer, checkpoint 
    def train(self, 
              cuda=False,
              lr=0.01,
              model_name="cl-tohoku/bert-base-japanese-whole-word-masking"):
        x = str(input("It takes about an hour to process, is that okay? (yes/no)"))
        if x in ["yes", "y", "Y"]:
            trainer, checkpoint = self._trainer(cuda)
            model = BertForMaskedLM_pl(model_name, lr)
            trainer.fit(model, self.dataloader_train, self.dataloader_val)
            best_model_path = checkpoint.best_model_path
            print(f"best_model_path : {best_model_path}")
            self._test(best_model_path)
        else:
            print("stopping ...")
    def _test(self, best_model_path: str):
        num = 0
        for data in tqdm(self.dataset_test):
            wrong = data["wrong_text"]
            correct = data["correct_text"]
            predict_text = predict.predict(wrong, best_model_path)
            # 文章がすべて正解していたら加算する
            if correct == predict_text:
                num += 1
        print(f"accuracy test: {num/len(self.dataset_test):.2f}")
        

trainer = Trainer()
print("Start loading the dataset ..................................")
trainer.load_dataloader(max_length=max_length, batch_size=batch_size)
print("Dataset loading is complete !!!")
print("Then start learning .......................................")
trainer.train(lr=lr) # gpuを指定する場合は引数に追記する