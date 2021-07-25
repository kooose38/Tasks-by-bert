from transformers import BertForMaskedLM 
import torch 
def BertForMaskedLM_(input_ids):
    """
    (文章数, 単語数, 1) input-layers
      -> (文章数, 単語数, 768) BertModel
        -> (文章数, 単語数, 32000) Linear/GReLU output-layers
    """
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    bert_mlm = BertForMaskedLM.from_pretrained(model_name)
    
    with torch.no_grad():
        output = bert_mlm(input_ids=input_ids)
        scores = output.logits 
    return scores 