from trainer import Trainer 
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("-l",
                   "--max_length",
                   help="token length",
                   type=int,
                   default=128)
parser.add_argument("-b",
                   "--batch_size",
                   help="batch size by train dataset",
                   type=int,
                   default=32)
parser.add_argument("-r",
                   "--lr",
                   help="learning rate",
                    type=float,
                   default=0.1)
parser.add_argument("-e",
                   "--epoch",
                   help="epoch",
                   type=int,
                   default=10)
parser.add_argument("-c",
                   "--cuda",
                   help="use gpu device",
                   type=bool,
                   choices=[True, False],
                   default=False)

args = parser.parse_args()

trainer = Trainer()

trainer.train(args.max_length, 
              args.cuda,
              args.batch_size,
              args.lr,
              args.epoch)