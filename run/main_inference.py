from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
import data.retrive_data_balanced as retrive_data
import utils.params as params
import nns.generator as generator
import nns.encoder as encoder
import run.inference as inference 
import numpy as np
import random
import os
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = params.parse_args()
    args.g_path = os.path.join(args.save_dir, "g_"+os.path.basename(args.model_form)+"_"+args.id)
    args.f_path = os.path.join(args.save_dir, "f_"+os.path.basename(args.model_form)+"_"+args.id)
    args.result_path = os.path.join(args.results_dir, os.path.basename(args.model_form)+"_"+args.id+"_inference")
    args.gumbel_t = args.init_t
    args.lr = args.init_lr
    set_seed(args.rand_seed)
    _, _, test_data = retrive_data.get_dataloaders(args)
    gen, enc = generator.Generator(args), encoder.Encoder(args)
    gen.load_state_dict(torch.load(args.g_path+str(args.rand_seed), weights_only=True))
    enc.load_state_dict(torch.load(args.f_path+str(args.rand_seed), weights_only=True))
    gen.eval()
    enc.eval()
    if args.test:
        test_stats = inference.test(test_data, gen, enc, args)
    #with open(args.result_path, "a") as f:
    #    f.write("\n")
