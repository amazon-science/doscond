from math import ceil
import torch
import torch.nn.functional as F
from graph_agent import GraphAgent
import argparse
import random
import numpy as np
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='DD')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--init', type=str, default='real')
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--nconvs', type=int, default=3)
parser.add_argument('--outer', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--lr_model', type=float, default=0.005)
parser.add_argument('--stru_discrete', type=int, default=1)
parser.add_argument('--ipc', type=int, default=0, help='number of condensed samples per class')
parser.add_argument('--reduction_rate', type=float, default=0.1, help='if ipc=0, this param  will be enabled')
parser.add_argument('--save', type=int, default=0, help='whether to save the condensed graphs')
parser.add_argument('--dis_metric', type=str, default='mse', help='distance metric')
parser.add_argument('--eval_init', type=int, default=1, help='whether to evaluate initialized graphs')
parser.add_argument('--bs_cond', type=int, default=256, help='batch size for sampling graphs')
parser.add_argument('--net_norm', type=str, default='none')
parser.add_argument('--beta', type=float, default=0.1, help='coefficient for the regularization term')
args = parser.parse_args()

if args.dataset == 'ogbg-molhiv':
    args.pooling = 'sum'
if args.dataset == 'CIFAR10':
    args.net_norm = 'instancenorm'
if args.dataset == 'MUTAG' and args.ipc == 50:
    args.ipc = 20
torch.cuda.set_device(args.gpu_id)

# torch.set_num_threads(1)

print(args)
device = 'cuda'

data = Dataset(args)
packed_data = data.packed_data

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

agent = GraphAgent(data=packed_data, args=args, device=device, nnodes_syn=get_mean_nodes(args))
agent.train()
