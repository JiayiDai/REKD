#import torch
#import pdb
import argparse

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description='Rationale-Net Classifier')
    #setup
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    # learning
    parser.add_argument('--init_lr', type=float, default=1e-5, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=35, help='number of epochs for train [default: 35]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 32]')
    parser.add_argument('--patience', type=int, default=999, help='Num epochs of no dev progress before half learning rate [default: 999]')
    #paths
    parser.add_argument('--save_dir', type=str, default='saved', help='where to save the snapshot')
    parser.add_argument('--results_dir', type=str, default='results', help='where to dump model config and epoch stats. If get_rationales is set to true, rationales for the test set will also be stored here.')
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    # model
    parser.add_argument('--model_form', type=str, default='bert', help='model is cnn/bilstm/bert' )
    parser.add_argument('--hidden_dim', type=int, default=100, help="Dim of hidden layer")
    parser.add_argument('--num_layers', type=int, default=1, help="Num layers of model_form to use")
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 norm penalty [default: 0]')
    parser.add_argument('--filter_num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--filters', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    # bilstm
    parser.add_argument('--bilstm_dim', type=int, default=256, help='number of lstm nodes')
    # data
    parser.add_argument('--dataset', default='imdb', help='choose which dataset to run on. [default: imdb]')
    # gumbel
    parser.add_argument('--init_t', type=float, default=5, help="Start temperature for gumbel softmax")
    #parser.add_argument('--gumbel_decay', type=float, default=3e-4, help="Start temprature for gumbel softmax. This is annealed via linear decay")
    # rationale
    parser.add_argument('--get_rationales',  action='store_true', default=False, help="otherwise, just train encoder")
    parser.add_argument('--select_lambda', type=float, default=.001, help="y1 in Gen cost L + y1||z|| + y2|zt - zt-1| + y3|{z}|")
    parser.add_argument('--contig_lambda', type=float, default=0, help="y2 in Gen cost L + y1||z|| + y2|zt - zt-1|+ y3|{z}|")
    parser.add_argument('--target_sparsity', type=float, default=0.1, help='Target fraction of features to select')
    parser.add_argument('--total_features', type=float, default=256, help='The number of total features for selection')
    
    #experiments
    parser.add_argument('--rand_seed', type=int, default=2025, help="Random seed for torch")
    parser.add_argument('--warmup',  action='store_true', default=False, help="Linear LR warm up")
    #parser.add_argument('--datasize', type=int, default=5000, help="The number of examples used for training")
    parser.add_argument('--id', type=str, default="test_run", help="Give a name for result/model paths")
    parser.add_argument('--id_t', type=str, default="test_run", help="To indicate teacher model path")

    #kd
    parser.add_argument('--model_form_t', type=str, default='Not specified', help='teacher model' )
    parser.add_argument('--kd_r_lambda', type=float, default=.01, help="lambda for rationale distilling loss")
    parser.add_argument('--hard_label',  action='store_true', default=False, help="If using hard teacher label")
    parser.add_argument('--alpha_re', type=float, default=0.5, help="alpha_re if fixing alpha_re")
    parser.add_argument('--dist_part', type=str, default='both', help='what to distill, rationale or prediction or both' )
    args = parser.parse_args()
    # update args and print
    args.filters = [int(k) for k in args.filters.split(',')]

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args