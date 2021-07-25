from transformers import BertForSequenceClassification
import pytorch_lightning as pl

class BertForSequenceClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        """
        (文章数, 単語数, 1) 入力層
          -> (文章数, 単語数, 768) BertModel
            -> (文章数, 768) tokenから[CLS]のベクトルのみを抽出
              -> (文章数, num_labels) Linear/ ReLU/ Softmax
        """
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率

        super().__init__()
        self.save_hyperparameters() 

        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch) 
        loss = output.loss
        self.log('train_loss', loss) 
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss) 

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        self.log('accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)