from model2 import BertForSequenceClassification_
from dataloader import DataLoader_
import category 
import torch 
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self):
        
        self.model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        self.model_path = "" 
        self.category = category.get_category() 
        
    def _trainer(self, lr: float):
        
        loss_f = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        writer = SummaryWriter("runs/livedoor_experiment_1")
        return loss_f, optimizer, writer
    
    def train(self, max_length=32, cuda=False, batch_size=32, lr=0.01, epoch=5):
        # データセットの読み込み
        data = DataLoader_()
        train, val, test = data.dataloader(max_length, cuda, batch_size)
        # モデルの構築
        model = BertForSequenceClassification_(self.model_name, 
                                              len(self.category))
        if cuda:
            model = model.cuda()
        loss_f, optimizer, writer = self._trainer(lr)
        
        # 学習と検証
        epoch = epoch
        running_loss = 0.0 # 訓練データのバッチ当たりのloss
        for e in range(epoch)
            val_acc, val_loss = 0, 0 # epoch毎にリセット
            model.train()
            for i, x in enumerate(train):
                t = x["labels"]
                y = model(x)
                assert len(t) == len(y)
                
                loss = loss_f(y, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if i%100 == 99:
                    # tensorboardにtrain-lossの書き込み
                    writer.add_scaler("training loss",
                                     running_loss/100,
                                     e*len(train)*batch_size+i)
                    running_loss = 0.0
            # 検証ロス、正解率
            model.eval()
            for x_v in val:
                t_v = x_v["labels"]
                with torch.no_grad():
                    y_v = model(x_v)
                loss_v = loss_f(y_v, t_v)
                y_v = y_v.argmax(-1)
                acc = (y_v == t_v).sum().item()
                val_loss += loss_v 
                val_acc += acc 
                all_len += t_v.size()[0]
            print(f"epoch: {e+1} -- val Loss: {val_loss} -- val accuracy: {val_loss/all_len:.3f}")
            
        # モデルの保存
        self.model_path = "./model/model_weights.pth"
        torch.save(model.state_dict(). self.model_path)
        
        x = str(input("Is this running in ipython environment? (yes/no)"))
        if x in ["yes", "y", "Y"]: # .ipythonであればtensorboardの表示
            self._tensorboard()
            
        # 保存したモデルでテスト評価
        self._test(test, cuda)
        
    def _tensorboard(self):
        %tensorboard --logdir runs/livedoor_experiment_1
        
    def _test(self, test, cuda: bool):
        model = BertForSequenceClassification_(self.model_name, 
                                               len(self.category))
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        all_len, acc = 0, 0
        for x_t in test:
            t_t = x_t["labels"]
            with torch.no_grad():
                y_t = model(x_t)
            y_t = y_t.argmax(-1)
            acc += (t_t == y_t).sum().item()
            all_len += t_t.size()[0]
        print(f"test accuracy: {acc/all_len:.3f}")
            
        
        