import sys 
from code import predict
args = sys.argv 
text = args[1]
best_model_path = "./model/best_model.ckpt"

predict_text = predict.predict(text, best_model_path, cuda=False)
print("--")
print(f"input: {text}")
print(f"output: {predict_text}")