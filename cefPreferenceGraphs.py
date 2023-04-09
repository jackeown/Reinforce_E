from e_caller import ECallerHistory
from helpers import normalizeState
import argparse
import termplotlib as tpl
from rich import print
from rich.progress import track
import random
import torch

# This will create 20 plots.
# One for each CEF.
# Each plot contains mulitple lines (one for each proof attempt)

parser = argparse.ArgumentParser()
parser.add_argument("hist", type=ECallerHistory.load)
parser.add_argument("policy", type=torch.load)
args = parser.parse_args()

def plot(statess, policy, action):
	all_ys = []
	for states in statess[:100]:
		if len(states):
			states = normalizeState(torch.from_numpy(states).to(torch.float))
			soft = torch.softmax(policy(states), dim=1)
			dist = torch.distributions.Categorical(soft)
			ys = torch.exp(dist.log_prob(torch.tensor(action)))
			all_ys.append(ys)
	return all_ys



random.seed(0)

infos_list = sum(list(args.hist.history.values()), [])
random.shuffle(infos_list)

states_list = [info['states'] for info in infos_list]

torch.save([plot(states_list, args.policy, i) for i in track(range(20))], "to_plot.pt")

