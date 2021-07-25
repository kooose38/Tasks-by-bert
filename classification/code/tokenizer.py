from transformers import BertJapaneseTokenizer 

def tokenizer_():
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    return tokenizer 