from math import ceil
import torch
import torch.nn.functional as F
from graph_agent import GraphAgent
import argparse
import random
import numpy as np
from utils import *
import sys
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--init', type=str, default='noise')
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--reduction_rate', type=float, default=0.1)
parser.add_argument('--nconvs', type=int, default=3)
parser.add_argument('--outer', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--ipc', type=int, default=0)
parser.add_argument('--nruns', type=int, default=10)
parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
parser.add_argument('--num_blocks', type=int, default=1)
parser.add_argument('--num_bases', type=int, default=0)
parser.add_argument('--stru_discrete', type=int, default=1)
parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--net_norm', type=str, default='none')
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--filename', type=str)

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

torch.set_num_threads(1)

args.ipc = int(args.filename.split('_')[1][3:])
args.seed = int((args.filename.split('_'))[2][1:])

LOG_FILENAME = f'logs/{args.dataset}_seeds.log'
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

print(args)
device = 'cuda'

data = Dataset(args)
packed_data = data.packed_data

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset == 'ogbg-molhiv':
    args.pooling = 'sum'

if args.dataset == 'CIFAR10':
    args.nruns = 3
    args.net_norm = 'instancenorm'

agent = GraphAgent(data=packed_data, args=args, device=device, nnodes_syn=get_mean_nodes(args))
assert args.stru_discrete == 1, 'must be discrete'

if args.stru_discrete:
    agent.adj_syn, agent.feat_syn = torch.load(f'saved/{args.filename}', map_location='cuda')


res = []
for _ in range(args.nruns):
    if args.dataset in ['ogbg-molhiv']:
        res.append(agent.test(epochs=100))
    else:
        res.append(agent.test(epochs=500))


res = np.array(res)
print('Mean Train/Val/TestAcc/TrainLoss:', res.mean(0))
print('Std Train/Val/TestAcc/TrainLoss:', res.std(0))

logging.info(str(args)+'\n'+f'Mean Train/Val/TestAcc/TrainLoss: {res.mean(0)}')

