from bottle import route, run 
from predict import predict_from_text 
import sys 

args = sys.argv

@route("/")
def index():
    pred = predict_from_text(args[1], filepath="./model/model_weights.pth", flag=True)
    return pred

run(host="localhost", port=5551)
    