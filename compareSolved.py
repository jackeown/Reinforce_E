# Comparing two experiments...

import numpy as np
import os, sys
import torch
from e_caller import ECallerHistory

from rich import print
from rich.progress import track

# Toggle rich progress...
track = lambda l:l



hists = {x: ECallerHistory.load(x) for x in track(sys.argv[1:])}

mean = lambda l: sum(l) / len(l)
median = lambda l: l[len(l)//2] if len(l)%2 == 1 else mean(l[len(l)//2 - 1 : len(l)//2 + 1])


def getProcCounts(history):
	procCounts = {os.path.split(prob)[1]: history.getProcCounts(prob) for prob in history.history.keys()}
	procCounts = {k:mean(v) for k,v in procCounts.items() if len(v)}
	return procCounts

procCounts = {k: getProcCounts(v) for k,v in hists.items()}
probsSolved = {key: {k for k in procCounts[key]} for key in procCounts}

def makeRow(key):
	solved = probsSolved[key]
	row = []
	for i,k in enumerate(probsSolved):
		#row.append( len(solved.intersection(probsSolved[k])) )
		if solved == probsSolved[k]:
			s = "[green]" + f"{len(solved):<26}" + "[/green]"
		else:
			padding = f"{'':<13}" if i < len(probsSolved)-1 else ""
			s = "[white]" + f"{len(probsSolved[k])-len(solved):<6}" + f"({len(probsSolved[k] - solved):>5})" + padding + "[/white]"
		row.append(s)
	return row

rows = [makeRow(k) for k in track(probsSolved)]

print(" "*26, end='')
for k in probsSolved:
	print(f"[blue]{k:<26}[/blue]", end="")
print("")


for (i,row),key in zip(enumerate(rows),probsSolved.keys()):
	print(f"[blue]{key:<26}[blue]", end='')
	for j,x in enumerate(row):
		print(x, end='')
	print("")

