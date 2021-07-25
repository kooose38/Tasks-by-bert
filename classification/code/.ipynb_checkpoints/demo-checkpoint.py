from predict import predict_from_text 
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("text",
                   help="Please enter the sentence you want to predict",
                   type="str")
parser.add_argument("-f",
                   "--filepath",
                   help="The path of the file that stores the learned weights",
                   type="str",
                   default="./model/model_weights.pth")

args = parser.parse_args()

predict_from_text(args.text, args.filepath)