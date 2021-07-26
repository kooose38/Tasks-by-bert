from bottle import route, run 
from demo import pred 
from predict import PredictFromMaskedSentence
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("text", 
#                     nargs="*",
                    help="Please enter the sentence you want to predict",
                    type=str)
parser.add_argument("-m",
                    "--num_mask", 
                    help="Number to mask from text",
                    type=int, default=2)
parser.add_argument("-t",
                    "--num_topk",
                    help="How much of the top output", 
                    type=int, default=3)
parser.add_argument("-hh",
                    "--method",
                    choices=["greedy", "beam"], 
                    help="Select a method",
                    type=str, default="greedy")

args = parser.parse_args()

@route("/")
def index():
    return {"result": pred(args, flag=True)}

run(host="localhost", port=5553)
