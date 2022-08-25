import os.path as osp
from math import ceil
import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DenseDataLoader
import torch.nn.functional as F
from graph_agent import GraphAgent
import argparse
import random
from utils import *
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='DD')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--nconvs', type=int, default=3)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--init', type=str, default='noise')
parser.add_argument('--lr_adj', type=float, default=1)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--reduction_rate', type=float, default=1)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--nruns', type=int, default=10)
parser.add_argument('--ipc', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--net_norm', type=str, default='none')
parser.add_argument('--stru_discrete', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
torch.set_num_threads(1)

# random seed setting
data_seed = 0
random.seed(data_seed)
np.random.seed(data_seed)
torch.manual_seed(data_seed)
torch.cuda.manual_seed(data_seed)

print(args)

data = Dataset(args)
packed_data = data.packed_data

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = 'cuda'
max_nodes = 1
agent = GraphAgent(data=packed_data, args=args, device=device, nnodes_syn=max_nodes)
train_dataset = packed_data[0]
sampled = []
for c in range(train_dataset.num_classes):
    ind = agent.syn_class_indices[c]
    idx_shuffle = np.random.permutation(agent.real_indices_class[c])[:ind[1]-ind[0]]
    sampled.append(agent.data[4][idx_shuffle])
agent.adj_syn = np.hstack(sampled)


runs = args.nruns
res = []
for _ in tqdm(range(runs)):
    if args.dataset in ['ogbg-molhiv']:
        res.append(agent.test_pyg_data(save=args.save, epochs=100, verbose=0))
    else:
        res.append(agent.test_pyg_data(save=args.save, epochs=500, verbose=0))

res = np.array(res)
print('Mean Train/Val/TestAcc/TrainLoss:', repr(res.mean(0)))
print('Std Train/Val/TestAcc/TrainLoss:', repr(res.std(0)))

