import argparse
from glob import glob
import os
from rich.progress import track
from rich import print
from main import runE
from e_caller import loadConfigurations
import torch
import functools
import multiprocessing as mp


# track = lambda x: x # For debugging...


def config2CEFs(config, scheduleConfigs):
    return scheduleConfigs[config]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eproverPath", default="eprover-ho")
    parser.add_argument("--eproverScheduleFile", default=os.path.expanduser("~/eprover/eprover_MASTER/HEURISTICS/schedule.vars"))
    parser.add_argument("--problemsPath", default=os.path.expanduser("~/Desktop/ATP/GCS/SLH-29/"))
    parser.add_argument("--results", default="extractedCEFs.pt")
    args = parser.parse_args()

    problems = [p for p in glob(f"{args.problemsPath}/**/*.p", recursive=True) if "Folds" not in p]


    scheduleConfigs = loadConfigurations(args.eproverScheduleFile)
    print(f"Schedule file contains {len(scheduleConfigs)} schedules")
    print(scheduleConfigs)


    configs = []
    cefs = []

    print(f"Attempting to solve {len(problems)} problems using --auto")
    
    runner = functools.partial(runE, None, args.eproverPath, None, auto=True, create_info=False)

    with mp.Pool(100) as p:
        results = p.map(runner, problems)
        for stdout, stderr in results:
            presat = True
            for line in stderr.split("\n"):
                if "configuration" in line.lower():
                    if not presat:
                        config = line.split(" ")[2]
                        configs.append(config)
                        cefs.append(config2CEFs(config, scheduleConfigs))
                    else:
                        presat = False


    torch.save(cefs, args.results)
