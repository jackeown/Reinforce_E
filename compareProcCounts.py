# Comparing two experiments...

import numpy as np
from scipy.stats import gmean
from rich import print
import os, sys
import torch
from e_caller import ECallerHistory
import IPython

np.seterr(invalid='ignore')

hists = {x: ECallerHistory.load(x) for x in sys.argv[1:]}

def tolerant_round(x):
	if not any([isinstance(x,t) for t in [int,float]]) or np.isnan(x):
		return x
	return int(x)


def getProcCounts(history, meaned=True):
	mean = lambda l: sum(l) / len(l)
	procCounts = {os.path.split(prob)[1]: history.getProcCounts(prob) for prob in history.history.keys()}
	procCounts = {k:v for k,v in procCounts.items() if 0 not in v}
	if meaned:
		procCounts = {k:mean(v) for k,v in procCounts.items() if len(v)}
	return procCounts

procCounts = {run: getProcCounts(hist) for run,hist in hists.items()}
probsSolved = {run: set(procCounts[run].keys()) for run in procCounts}

mean = lambda l: sum(l) / len(l)
#gmean = lambda l: mean(l)
median = lambda l: l[len(l)//2] if len(l)%2 == 1 else mean(l[len(l)//2 - 1 : len(l)//2 + 1])

def procCountDiff(key1,key2):
	try:
		pc1, pc2 = procCounts[key1], procCounts[key2]
		diffs = [pc2[k] - pc1[k] for k in pc1 if k in pc2] # intersection
		return [mean(diffs), median(diffs), gmean(diffs)] if len(diffs) else [np.nan, np.nan, np.nan]
	except:
		IPython.embed()

def makeRow(key):
	row = []
	for k in probsSolved:
		if k == key:
			x = list(procCounts[k].values())
			row.append((mean(x),median(x),tolerant_round(gmean(x))))
		else:
			row.append(procCountDiff(key, k))
	return row


if __name__ == "__main__":
	# IPython.embed()

	# Make sure that each of these runs represents an "evaluation" and not a "training"
	for k in probsSolved:
		m = max([len(p) for p in getProcCounts(hists[k], meaned=False).values()])
		if m>1:
			print("Max len(procCounts) for {k}: {m}")


	#	print("How to interpret: mean (median) [geometric mean]")
	# Print Column Headers
	print(" "*26, end='')
	for k in probsSolved:
		print(f"[blue]{k:<26}[/blue]", end="")
	print("")


	# Print Rows Themselves
	rows = [makeRow(k) for k in probsSolved]
	for (i,row),key in zip(enumerate(rows),probsSolved.keys()):
		print(f"[blue]{key:<26}[/blue]", end='')
		for j,(x,y,z) in enumerate(row):
			x=tolerant_round(x)
			y=tolerant_round(y)
			#z=int(z)
			s = f"{x:<6} ({y:>5}) [{z:>5}]{'':<4}"
			if i==j:
				s = "[green]"+s+"[/green]"
			else:
				s = "[white]"+s+"[/white]"
			print(s, end='')
		print("")
