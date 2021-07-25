from predict import PredictFromMaskedSentence
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("text", 
                    help="Please enter the sentence you want to predict", type=str)
parser.add_argument("-m", "--num_mask", help="Number to mask from text", type=int, default=2)
parser.add_argument("-t", "--num_topk", help="How much of the top output", type=int, default=3)
parser.add_argument("-hh", "--method", choices=["greedy", "beam"], help="Select a method", type=str, default="greedy")

args = parser.parse_args()

pred = PredictFromMaskedSentence()
if args.method == "greedy":
    print("We will solve the fill-in-the-blank problem with greedy-prediction")
    print("~"*100)
    pred.greedy_prediction(args.text, args.num_mask, args.num_topk)
elif args.method == "beam":
    print("We will solve the fill-in-the-blank problem with beam-search")
    print("~"*100)
    pred.beam_search(args.text, args.num_mask, args.num_topk)

