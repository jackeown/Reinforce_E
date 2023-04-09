import subprocess
import sys

for i in range(5):
	runs = [f"{run}{i}" for run in sys.argv[1:]]
	command = "python compareProcCounts.py " + " ".join(runs)
	subprocess.run(command, shell=True)
