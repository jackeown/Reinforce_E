from e_caller import ECallerHistory
import sys
from rich.progress import track
from rich import print
import IPython

hist = ECallerHistory.load(sys.argv[1])

searchFor = "Processed" if sys.argv[2].lower().startswith("p") else "Unprocessed"
searchFor = f"{searchFor} tracking failure"

for key, l in track(hist.history.items(), description=f"Looking for '{searchFor}'"):
	for x in l:
		if searchFor in x['stdout']:
			print(x['stdout'])
			sys.exit()


IPython.embed()
