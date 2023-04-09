# Comparing two experiments...

import numpy as np
import os, sys
import torch
from e_caller import ECallerHistory

rr = ECallerHistory.load(sys.argv[1])
rl = ECallerHistory.load(sys.argv[2])

def getProcCounts(history):
	mean = lambda l: sum(l) / len(l)
	procCounts = {os.path.split(prob)[1]: history.getProcCounts(prob) for prob in history.history.keys()}
	procCounts = {k:mean(v) for k,v in procCounts.items() if len(v)}
	return procCounts

rrProcCounts = getProcCounts(rr)
rlProcCounts = getProcCounts(rl)


rrProbsSolved = {k for k in rrProcCounts}
rlProbsSolved = {k for k in rlProcCounts}

bothSolved = rrProbsSolved.intersection(rlProbsSolved)
eitherSolved = rrProbsSolved.union(rlProbsSolved)


l = []
for prob in rrProbsSolved.intersection(rlProbsSolved):
	l.append(rrProcCounts[prob] - rlProcCounts[prob])


print(f"only {sys.argv[1]} solved  {len(rrProbsSolved-rlProbsSolved)}")
print(f"only {sys.argv[2]} solved {len(rlProbsSolved-rrProbsSolved)}")
print(f"Both solved {len(bothSolved)}")


print(f"{np.median(l)} fewer clauses processed by {sys.argv[2]} on average!")
