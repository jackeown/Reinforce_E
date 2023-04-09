from e_caller import ECallerHistory
import sys
from rich.progress import track
from rich import print
import functools

def getArgs(hist):
	lists_of_infos = list(hist.history.values())
	for infos in lists_of_infos:
		for info in infos:
			return info['args']

if __name__ == "__main__":
	hists = [ECallerHistory.load(x) for x in track(sys.argv[1:])]

	for hist, name in zip(hists, sys.argv[1:]):
		print(f"[blue]{name}: [/blue]")
		print(getArgs(hist))
		print("\n" + "-"*80 + "\n")
