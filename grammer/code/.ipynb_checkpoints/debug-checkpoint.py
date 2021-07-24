# import sys 
# import predict 
# args = sys.argv 
# text = args[1]
# best_model_path = "./model/###"

# predict_text = predict.predict(text, best_model_path)
# print("--")
# print(f"input: {text}")
# print(f"output: {predict_text}")

import os 
print(os.environ.get("MODEL_NAME"))