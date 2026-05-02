from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
import data.retrive_data_balanced as retrive_data
import utils.params as params
import nns.generator as generator
import nns.encoder as encoder
import run.learn as learn 
import numpy as np
import random
import os

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = params.parse_args()
    set_seed(args.rand_seed)

    args.gt_path = os.path.join(args.save_dir, "g_"+os.path.basename(args.model_form_t)+"_"+args.id_t)
    args.ft_path = os.path.join(args.save_dir, "f_"+os.path.basename(args.model_form_t)+"_"+args.id_t)
    args.g_path = os.path.join(args.save_dir, "g_"+os.path.basename(args.model_form)+"_"+args.id)
    args.f_path = os.path.join(args.save_dir, "f_"+os.path.basename(args.model_form)+"_"+args.id)
    args.result_path = os.path.join(args.results_dir, os.path.basename(args.model_form)+"_"+args.id)
    
    args.gumbel_t = args.init_t
    args.lr = args.init_lr
    
    train_data, dev_data, test_data = retrive_data.get_dataloaders(args)

    gen_t, enc_t = generator.Generator(args, if_t=True), encoder.Encoder(args, if_t=True)
    gen_t.load_state_dict(torch.load(args.gt_path, weights_only=True), strict=False)
    enc_t.load_state_dict(torch.load(args.ft_path, weights_only=True), strict=False)
    gen_t.eval()
    enc_t.eval()
    gen_s, enc_s = generator.Generator(args), encoder.Encoder(args)
    if args.train:
        gen_s, enc_s = learn.train(train_data, dev_data, gen_s, enc_s, args, gen_t, enc_t)
    if args.test:
        test_stats = learn.test(test_data, gen_s, enc_s, args)
    with open(args.result_path, "a") as f:
        f.write("\n")
