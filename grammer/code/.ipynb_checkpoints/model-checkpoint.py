import torch
from transformers import BertForMaskedLM
import pytorch_lightning as pl
class BertForMaskedLM_pl(pl.LightningModule):
        
    def __init__(self, model_name: str, lr: float):
        """
        token毎に32000次元のクラス分類を行う。
        32000はbertに登録されているボキャブラリー数のことを指す。
        
        (文章数, 単語数, 1) 入力層
          -> (文章数, 単語数, 32000) 出力層
            -> (文章数, 単語数, 1) 32000から最大の確率である`index`を取得し、bertの辞書から当てはまる単語を予測する
        """
        super().__init__()
        self.save_hyperparameters()
        self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)
        
    def training_step(self, batch, batch_idx):
        output = self.bert_mlm(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_mlm(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)
   
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)