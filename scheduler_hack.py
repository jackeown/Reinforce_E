# I understand this file and overall approach is a nightmare.
# It is, however, the only way I could think of to have the 
# e_caller.py instances cooperate without complicating things even further...please forgive me.

import os
from time import sleep
from glob import glob

folder = "SCHEDULER_HACK"

probId = lambda p: int(p[-8:-4])
numToPath = lambda num: f"./{folder}/{num}"

addProblem = lambda x: open(numToPath(x), 'a').close()
removeProblem = lambda x: os.remove(numToPath(x))
checkProblemExists = lambda x: os.path.exists(numToPath(x))

anyLeftToProve = lambda setOfProblems: any([checkProblemExists(probId(p)) for p in setOfProblems])


if __name__ == "__main__":
    os.makedirs(folder, exist_ok=True)
    for x in glob(f"{folder}/*"):
        os.remove(x)


    while True:

        print("Adding...", end='')
        for x in range(1,2079):
            addProblem(x)
        print("Added")
        
        remaining = glob(f"{folder}/*")
        while remaining:
            print(f"Waiting...{len(remaining)}")
            sleep(1 if len(remaining) < 200 else 10)
            remaining = glob(f"{folder}/*")


