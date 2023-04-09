# Simple wrapper around runE function from main.py so that it's easy to run E with a learned policy...
import argparse
import torch
from main import runE
import IPython


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("--policy", default="latest_model.pt")
    parser.add_argument("--cef_file", default="cefs_auto.txt")
    parser.add_argument("--cpu-limit", default=10, type=int)
    parser.add_argument("--create_info", action="store_true")
    args = parser.parse_args()

    policy = torch.load(args.policy)

    with open(args.cef_file) as f:
        CEF_STRING = f.read()

    info = runE(policy, "eprover_RL", CEF_STRING, args.problem, 
        max(args.cpu_limit-5, 5), args.cpu_limit,
        auto=False, create_info=args.create_info, verbose=True)
    

    if args.create_info:
        IPython.embed()