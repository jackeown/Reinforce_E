from main import runE
import argparse
from glob import glob
import random
from os.path import expanduser
from policy_grad import PolicyNet, PolicyNetConstCategorical, PolicyNetRoundRobin, PolicyNetVeryBad
from time import sleep
import torch
import IPython

USER = expanduser("~")
CEF_STRING = open(expanduser(f"{USER}/Desktop/Reinforce E/cefs_auto_slh.txt")).read()


def tryToSolveProbAuto(problem):
    return runE(None, "eprover-ho",  None, problem, 20, 30, True, create_info=False, verbose=True)

def tryToSolveProbDummyModel(problem):
    # dummy = PolicyNetConstCategorical(5,20)
    dummy = PolicyNetRoundRobin(5,20)
    # dummy = PolicyNetVeryBad(20)
    return runE(dummy, "eprover_RL-ho",  CEF_STRING, problem, 20, 30, False, create_info=False, verbose=True)

def tryToSolveProbRoundRobin(problem):
    return runE(None, "eprover-ho", CEF_STRING, problem, 20, 30, False, create_info=False, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--problem", default="")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    args.problem = args.problem if args.problem != "" else random.choice(glob(f"{USER}/Desktop/ATP/GCS/SLH-29/Folds/0/train/*.p"))

    if args.auto:
        print(f"Solving {args.problem} with eprover-ho --auto" + "\n" + "#"*80)
        sleep(2)
        info = tryToSolveProbAuto(args.problem)
    elif args.rl:
        print(f"Solving {args.problem} with eprover_RL-ho" + "\n" + "#"*80)
        sleep(2)
        info = tryToSolveProbDummyModel(args.problem)
    else:
        print(f"Solving {args.problem} with eprover-ho" + "\n" + "#"*80)
        sleep(2)
        info = tryToSolveProbRoundRobin(args.problem)
    
    if args.prompt:
        IPython.embed()