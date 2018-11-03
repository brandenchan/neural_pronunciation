""" This file creates a CharToPhonModel that takes user input
and returns a prediction """

from model import CharToPhonModel
import argparse
import json

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', type=str)
args = parser.parse_args()

# Look in directory for "hyperparameters.json"
hyperparams_file = args.dir
if hyperparams_file[-1] != "/":
    hyperparams_file += "/"
hyperparams_file += 'results/hyperparameters.json'
settings = json.load(open(hyperparams_file))

m = CharToPhonModel(**settings)
m.interactive()

print("OK")

# if args.mode == "train":
#     m.train()
# elif args.mode == "validation":
#     m.validation()
# elif args.mode == "test":
#     m.test()
# elif args.mode == "all":
#     m.train()
#     m.validation()
#     m.test()
# else:
#     raise Exception

