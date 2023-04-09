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

# This will create 1 plot for each feature of the state.
# For each value, x, of a particular feature, f, we query all episodes for states with a value for f that is "close" to x.
# These states are queried for values, and the average value is used as a y value for each plot.

parser = argparse.ArgumentParser()
parser.add_argument("hist")
parser.add_argument("policy")
parser.add_argument("--max_eps", type=int, default=1000)
args = parser.parse_args()

hist,policy = args.hist, os.path.split(args.policy)[1]

args.hist = ECallerHistory.load(args.hist)
args.policy = torch.load(args.policy)

def plotInfo(allStates, policy, feature):
	all_ys = []
	allStates = torch.cat(allStates, dim=0)
	vals = allStates[:,feature]
	normedStates = normalizeState(allStates)
	stateValues = policy.critic(normedStates)

	low = vals.min().item()
	#high = [3000,1000,20000,40,50][feature]
	high = [1500, 500, 3000, 30,40][feature]


	dx = (high-low) / 20
	groupedValues = [[] for a in np.arange(low, high, dx)]
	for x, value in zip(vals, stateValues):
		x = x.item()
		try:
			bin = math.floor((x-low) / dx)
			if bin < len(groupedValues):
				groupedValues[bin].append(value.item())
		except:
			print(math.floor((x-low)/dx), len(groupedValues))
			IPython.embed()

	xs = list(np.arange(low,high,dx))
	ys = [SAVED:=(mean(vals), np.std(vals)) if len(vals) else (SAVED[0],1e-10) for i,vals in enumerate(groupedValues)]
	#ys = [(mean(vals), np.std(vals)) for i,vals in enumerate(groupedValues)]
	ys, yerrs = zip(*ys)
	return xs[:-1],ys[:-1],yerrs[:-1]


random.seed(0)

infos_list = sum(list(args.hist.history.values()), [])
random.shuffle(infos_list)

states_list = [torch.from_numpy(info['states']).to(torch.float) for info in infos_list if len(info['states'])]
stuff = [plotInfo(states_list[:args.max_eps], args.policy, feature) for feature in track(range(5))]


fig,axes = plt.subplots(5)
plt.tight_layout()

i = 0
for ax,(xs,ys,yerrs) in zip(axes,stuff):
	xs,ys,yerrs = np.array(xs), np.array(ys), np.array(yerrs)
	ax.plot(xs,ys, label=f"Feature {i}")
	ax.fill_between(xs, ys - yerrs, ys + yerrs, color='b', alpha=0.1)
	i += 1

fig.savefig(f"figures/critic_visualization_{hist}_{policy}.png")
