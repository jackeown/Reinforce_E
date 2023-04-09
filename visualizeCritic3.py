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



colors = ["red", "orange","yellow","green","blue"]

# This will create 1 plot with 6 lines: 1 for each feature and 1 for the critic evaluation

parser = argparse.ArgumentParser()
parser.add_argument("hist")
parser.add_argument("policy")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--count", type=int, default=1)
parser.add_argument("--max_len", type=int, default=20000)
parser.add_argument("--min_len", type=int, default=1000)

parser.add_argument("--failure", action='store_true')
parser.add_argument("--interactive", action='store_true')
parser.add_argument("--average", action="store_true")
parser.add_argument("--horizontal", action="store_true")
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


if args.horizontal:
    fig, axess = plt.subplots(1,6, sharex="all")
else:
    fig, axess = plt.subplots(6, sharex="all")

plt.tight_layout()


mean = lambda l: sum(l)/len(l)

if args.average:
    infos = [random.choice(infos_list) for _ in range(args.count)]
    statess = [torch.from_numpy(info['states']).to(torch.float) for info in track(infos)]
    valuess = [args.policy.critic(normalizeState(s)) for s in track(statess)]
    max_n = max([len(x['states']) for x in infos])

    states_avg = torch.stack([mean([s[i] for s in statess if i < s.shape[0]]) for i in track(range(max_n))])
    values_avg = torch.stack([mean([v[i] for v in valuess if i < v.shape[0]]) for i in track(range(max_n))])
    print("States_avg shape", states_avg.shape)

    for i in range(5):
        ys = states_avg[:,i].reshape(-1).detach().numpy()
        axess[i].plot(ys, color=colors[i])
    
    axess[-1].plot(values_avg.detach().numpy(), color="black")

else:
    for _ in track(range(args.count)):
        info = random.choice(infos_list)

        states = torch.from_numpy(info['states']).to(torch.float)
        for i in range(5):
            axess[i].plot(states[:,i].reshape(-1).detach().numpy(), color=colors[i])
        values = args.policy.critic(normalizeState(states)).detach().numpy()
        axess[-1].plot(values, label="Values", color="black")


for ax, title in zip(axess, ["Clauses Processed","len(proc)","len(unproc)","weight(proc)","weight(unproc)", "Critic Value"]):
    ax.set_title(title)

if args.interactive:
    plt.show()

else:
    plt.savefig(f"figures/critic_visualization3_{policy}_{info['problem']}.png", dpi=500)
