from tokenzier import tokenzier_ 
from transfomers import BertJapaneseTokenizer 
import torch 

def predict_from_text(text: str, filepath: str):
    model = BertJapaneseTokenizer(3)
    model.load_state_dict(torch.load(filepath,
                                    map_location=torch.device("cpu")))
    tokenizer = tokenzier_()
    
    encoding = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(encoding)
        output = output[0].cpu().numpy()
        score = np.where(output >= .5, 1, 0).astype(int).tolist()
    print("--")
    print(f"input: {text}")
    category = {
        "negative": score[0],
        "neutral": score[1],
        "positive": score[2]
    }
    print(f"output: {category}")
    

sample = "去年から黒字が減少した"
predict_from_text(sample. "model_weights.pth")