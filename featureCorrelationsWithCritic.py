from e_caller import ECallerHistory
from helpers import normalizeState, mean
import argparse
import matplotlib.pyplot as plt
from rich import print
from rich.progress import track
import random
import torch
from collections import defaultdict
import numpy as np
import math
import os
import IPython
from rich.progress import track




parser = argparse.ArgumentParser()
parser.add_argument("hist")
parser.add_argument("policy")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--count", type=int, default=1)
parser.add_argument("--max_len", type=int, default=20000)
parser.add_argument("--min_len", type=int, default=1000)

parser.add_argument("--failure", action='store_true')
args = parser.parse_args()
random.seed(args.seed)


# For generated image filename
hist,policy = args.hist, os.path.split(args.policy)[1]

# load ECallerHistory and policy/critic
args.hist = ECallerHistory.load(args.hist)
args.policy = torch.load(args.policy)


# Choose a proof attempt
infos_list = sum(list(args.hist.history.values()), [])

accept = lambda info: (not info['solved']) if args.failure else info['solved']
infos_list = [x for x in infos_list if accept(x) and len(x['states']) > args.min_len and len(x['states']) < args.max_len]
print(f"{len(infos_list)} problems to sample from")


mean = lambda l: sum(l)/len(l)


correlations = []
for _ in track(range(args.count)):
    info = random.choice(infos_list)
    states = torch.from_numpy(info['states']).to(torch.float).reshape(-1,5)
    values = args.policy.critic(normalizeState(states)).reshape(-1,1)

    X = torch.cat([states,values], dim=1)
    
    print(X.shape)
    correlation = torch.corrcoef(X.T)
    correlations.append(correlation)
    print(correlation)


print("#"*100)
print(torch.stack(correlations).mean(dim=0))
