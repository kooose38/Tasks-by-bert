import sys 
from predict import predict_from_text 

args = sys.argv
predict_from_text(args[1], "./model/model_weights.pth")