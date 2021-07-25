from model2 import BertForSequenceClassification_
from category import get_category 
from tokenizer import tokenizer_
import torch 

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"

def predict_from_text(text: str, weight_file: str):
    # import modules
    category = get_category()
    model = BertForSequenceClassification_(model_name, len(category))
    model.load_state_dict(torch.load(weight_file,
                                     map_location=torch.device("cpu")))
    tokenizer = tokenizer_()
    
    encoding = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        scores = model(encoding)
        predict = scores.argmax(-1).item()
    
    print(f"input: {text}")
    print(f"output: {category[predict]}")
    