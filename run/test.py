from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
import data.retrieve_data_balanced as retrieve_data
import utils.params as params
import nns.generator as generator
import nns.encoder as encoder
import run.learn as learn 
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
    args.gt_path = os.path.join(args.save_dir, "g_"+os.path.basename(args.model_form_t)+"_"+args.id_t)
    args.ft_path = os.path.join(args.save_dir, "f_"+os.path.basename(args.model_form_t)+"_"+args.id_t)
    args.g_path = os.path.join(args.save_dir, "g_"+os.path.basename(args.model_form)+"_"+args.id)
    args.f_path = os.path.join(args.save_dir, "f_"+os.path.basename(args.model_form)+"_"+args.id)

    args.result_path = os.path.join(args.results_dir, os.path.basename(args.model_form)+"_"+args.id)
    args.gumbel_t = args.init_t
    args.lr = args.init_lr
    set_seed(args.rand_seed)
    train_data, dev_data, test_data = retrieve_data.get_dataloaders(args)

    gen_t, enc_t = generator.Generator(args, if_t=True), encoder.Encoder(args, if_t=True)
    gen_t.load_state_dict(torch.load(args.gt_path, weights_only=True))
    enc_t.load_state_dict(torch.load(args.ft_path, weights_only=True))
    gen_t.eval()
    enc_t.eval()
    gen_s, enc_s = generator.Generator(args), encoder.Encoder(args)
    gen_s.load_state_dict(torch.load(args.g_path+str(args.rand_seed), weights_only=True))
    enc_s.load_state_dict(torch.load(args.f_path+str(args.rand_seed), weights_only=True))
    gen_s.eval()
    enc_s.eval()
    if args.test:
        test_stats = learn.test(test_data, gen_s, enc_s, args, gen_t, enc_t)
    with open(args.result_path, "a") as f:
        f.write("\n")
