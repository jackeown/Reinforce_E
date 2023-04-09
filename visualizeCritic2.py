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

# This will create 1 plot for a random (seeded) successful proof attempt (or failure with --failure)

parser = argparse.ArgumentParser()
parser.add_argument("hist")
parser.add_argument("policy")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--samples", type=int, default=20)
args = parser.parse_args()
random.seed(args.seed)


# For generated image filename
hist,policy = args.hist, os.path.split(args.policy)[1]

# load ECallerHistory and policy/critic
args.hist = ECallerHistory.load(args.hist)
args.policy = torch.load(args.policy)


# Choose a proof attempt
infos_list = sum(list(args.hist.history.values()), [])

plt.rcParams['axes.facecolor'] = 'grey'

valuess = []
for failure in [False, True]:
	accept = lambda info: info['solved'] if not failure else not info['solved']
	correct_class = [info for info in infos_list if accept(info) and len(info['states']) > 5]
	for i in range(args.samples):
		info = random.choice(correct_class)
		states = torch.from_numpy(info['states']).to(torch.float)
		values = args.policy.critic(normalizeState(states)).detach().numpy()

		valuess.append((values, 'black' if failure else 'white'))

random.shuffle(valuess)
for values, color in valuess:
	plt.plot(values, color=color, alpha=0.2)

plt.xlim(0,1000)

plt.savefig(f"figures/critic_visualization2_{hist}_{policy}_{args.seed}b.png", dpi=1000)
