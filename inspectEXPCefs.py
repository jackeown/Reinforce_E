from e_caller import ECallerHistory, loadConfigurations
import argparse
from rich.progress import track
from rich import print
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--config", action="store_true")
args = parser.parse_args()

print("Loading History...", end='')
hist = ECallerHistory.load(args.run)
print("Loaded")


configurations = loadConfigurations("/Data/home/jack/eprover/eprover_MASTER/HEURISTICS/schedule.vars")

allCounts = defaultdict(lambda:0)
for prob, infos in track(hist.history.items()):
	for info in infos:
		if args.config:
			allCounts[info['configName']] += 1
		else:
			for cef, count in configurations[info['configName']].items():
				allCounts[cef] += int(count)
			#for action in info['actions']:
			#	allCounts[list(info['cefs'].keys())[action]] += 1



counts = sorted(list(allCounts.values()))
for thing,count in sorted(allCounts.items(), key=lambda x: x[1]):
	#print(thing, count)
	print(f"{round(count/216)}*{thing},")


import matplotlib.pyplot as plt
plt.plot(counts)
plt.show()

