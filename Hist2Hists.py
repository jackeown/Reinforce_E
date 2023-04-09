# The point of this script was to understand how to best normalize RL state
# Mostly irrelevant now, but I don't want to delete it because I'm a digital hoarder


from e_caller import ECallerHistory
import sys
import torch
from rich import print
import termplotlib as tpl
import numpy as np


hist = ECallerHistory.load(sys.argv[1])

allState = [torch.tensor(info['states'],dtype=torch.float) for infos in hist.history.values() for info in infos]
T = torch.cat(allState,dim=0)


for nums in T.T:
	counts, bin_edges = np.histogram(nums, bins=40)
	fig = tpl.figure()
	fig.hist(counts, bin_edges, grid=[15, 25], orientation="horizontal", force_ascii=True)
	fig.show()
	print("")

