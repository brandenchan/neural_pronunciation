""" This file creates a CharToPhonModel and performs training,
validation or testing. """

from model import CharToPhonModel
import argparse

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--validation', dest='mode', action='store_const', const="validation", default="all")
parser.add_argument('--train', dest='mode', action='store_const', const="train")
parser.add_argument('--test', dest='mode', action='store_const', const="test")

parser.add_argument('--debug', dest='debug', action='store_true', default=False)
args = parser.parse_args()

debug_settings = {"data_dir":"data/",
                "batch_size":4,
                "embed_dims":200,
                "hidden_dims":200,
                "bidir":False,
                "max_gradient_norm":1,
                "learning_rate":0.001,
                "save_dir":"unsaved_model/",
                "resume_dir":None,
                "n_batches":501,
                "debug":True,
                "print_every":10,
                "save_every":100,
                "cell_class":"lstm",
                "initializer":"glorot",
                "attention":"luong",
                "dropout":0.0,       
                "anneal_steps":1000,   
                "anneal_decay":0.95 
}

test_settings = {"data_dir":"data/",
                "batch_size":128,
                "embed_dims":300,
                "hidden_dims":300,
                "bidir":True,
                "max_gradient_norm":1,
                "learning_rate":0.001,
                "save_dir":"unsaved_model/",
                "resume_dir":None,
                "n_batches":10001,
                "debug":False,
                "print_every":100,
                "save_every":500,
}

if args.debug:
    settings = debug_settings
else:
    settings = test_settings

m = CharToPhonModel(**settings)

if args.mode == "train":
    m.train()
elif args.mode == "validation":
    m.validation()
elif args.mode == "test":
    m.test()
elif args.mode == "all":
    m.train()
    m.validation()
    m.test()
else:
    raise Exception

