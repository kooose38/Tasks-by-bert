from tokenizer import SC_tokenizer 
import torch 
from model import BertForMaskedLM_pl

def predict(text: str, best_model_path: str) -> str:
    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = SC_tokenizer.from_pretrained(MODEL_NAME)
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors="pt"
    )
    model = BertForMaskedLM_pl.from_checkpoint(
        best_model_path
    )
#     model = BertForMaskedLM_pl(MODEL_NAME, 0.01) ## debug用
    bert_mlm = model.bert_mlm
    with torch.no_grad():
        output = bert_mlm(**encoding)
        scores = output.logits # (1, 32, 32000)
        labels_predict = scores[0].argmax(-1).cpu().numpy().tolist() # (1, 32, 1)
    # bertにより予測されたトークンから文章を出力
    predict_text = tokenizer.convert_bert_output_to_text(
        text, labels_predict, spans
    ) 
    return predict_text
