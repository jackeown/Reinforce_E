import torch
import argparse
import re



def getCEFs(filename):
    with open(filename) as f:
        text = f.read()
    return re.findall(r"[0-9]+\*(.*)", text)

def getPolicy(filename):
    policy = torch.load(filename)
    policy = torch.softmax(policy(torch.tensor([[0,0,0,0]])), dim=1)[0]

    # n is one over the smallest action probability greater than 1%
    n = 5 * 1/policy[policy>0.01].min()
    return [int(x.item()) for x in (policy*n).round()]

def writeCEFSchedule(cefs, policy, output):

    if output == "":
        print(policy)
        return

    with open(output, "w") as f:
        f.write("(")
        lines = []
        for cef, count in zip(cefs, policy):
            lines.append(f"{count}*{cef}")
        f.write("\n".join(lines))
        # f.write(")")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="latest_model.pt")
    parser.add_argument("--original", default="cefs_auto.txt")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cefs = getCEFs(args.original)
    policy = getPolicy(args.policy)
    writeCEFSchedule(cefs, policy, args.output)
