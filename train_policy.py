# This file will maintain a pytorch model policy for interacting with E.
# It has 3 primary responsibilities.

# 1.) It learns from the .xp files in `./episodes` using off-policy policy gradients
# 2.) It occasionally writes the latest model state_dict to disk for `gain_xp.py` to use when creating episodes.

import argparse
from email import policy
import torch
import os, time
from typing import Dict
from helpers import Episode
from glob import glob
from policy_grad import PolicyNet, PolicyNetVeryBad, PolicyNetAttn, PolicyNetUniform, PolicyNetConstCategorical, calculateReturns, optimize_step
from time import sleep
from rich.progress import track
import random
from queue import Queue
import termplotlib as tpl
from SerializationTesting import load_episode_numpy

from my_profiler import Profiler



# Here for toggling verbose output.
def track(l, *args, **kwargs):
    return l

def mean(l):
    return sum(l) / len(l)

def prune_episode(episodes, ep_path):
    # if random.random() < 0.25:
    del episodes[ep_path]
    os.remove(ep_path)


def load_episode(episode):
    return load_episode_numpy(episode)


def finished_writing(episode):
    dt = time.time() - os.path.getmtime(episode)
    return (dt > 20.0) and (os.stat(episode).st_size > 0)


def keepTraining(newEpisodes, windowSize):
    print(f"Count Since Novel Solution: {keepTraining.countSinceNovelSolution}")

    for ep in newEpisodes:
        if ep.problem not in keepTraining.problemsSolved:
            keepTraining.problemsSolved.add(ep.problem)
            keepTraining.countSinceNovelSolution = 0
        else:
            keepTraining.countSinceNovelSolution += 1
            
    return (keepTraining.countSinceNovelSolution < windowSize)


keepTraining.countSinceNovelSolution = 0
keepTraining.problemsSolved = set()


# Mutates episodes
def load_episodes(episodes: Dict[str, Episode]):
    with my_profiler.profile("Checking for episodes to load"):
        known_episodes = set(episodes.keys())
        all_episodes = set([ep for ep in glob("./episodes/*.episode") if finished_writing(ep)])
        new_episodes = all_episodes - known_episodes
        print(f"About to load {len(new_episodes)}")
    
    with my_profiler.profile("Loading episodes"):
        paths = new_episodes
        new_episodes = [load_episode(path) for path in new_episodes]
        for path,ep in zip(paths, new_episodes):
            episodes[path] = ep

    print(f"Loaded {len(new_episodes)} new episodes ({len(episodes)} total)")
    load_episodes.historicalRewardInfo.extend([1 if x.rewards[-1] else 0 for x in new_episodes])
    return new_episodes

load_episodes.historicalRewardInfo = []



def save_latest_policy(policy_net, optimizer, policy_path, optimizer_path):
    torch.save(policy_net, policy_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def applyMaxBlame(ep, max_blame):
    if ep.rewards[-1]:
        return ep
    return Episode(ep.problem, ep.states[:max_blame], ep.actions[:max_blame], ep.rewards[:max_blame], ep.policy_net)



def learn(episodes, policy_net, optimizer, losses, batch_size, backlog_size):

    MAX_BLAME = 30_000

    print("enter Learn")
    while len(episodes) > backlog_size:
        batch_eps = list(episodes.keys())
        random.shuffle(batch_eps)
        batch_eps = batch_eps[:batch_size]

        has_critic = hasattr(policy_net, "critic")
        batch = [episodes[k] for k in batch_eps if has_critic or episodes[k].rewards[-1]]

        if has_critic:
            batch = [applyMaxBlame(ep, MAX_BLAME) for ep in batch]

        with my_profiler.profile("episode -> batch"):
            s = torch.cat([torch.cat(episode.states) for episode in batch], dim=0)
            a = torch.cat([torch.cat(episode.actions) for episode in batch], dim=0)
            returnss = torch.cat([calculateReturns(policy_net, episode, False, my_profiler) for episode in batch], dim=0)
        
        with my_profiler.profile("optimization step"):
            loss = optimize_step(optimizer, policy_net, s, a, returnss, profiler=my_profiler, batch_size=batch_size)
            print(f"    Loss for batch of size {len(batch)}: {loss}")
            losses.append(loss)
        
        with my_profiler.profile("pruning"):
            for ep in batch_eps:
                prune_episode(episodes, ep)
    
    print("exit Learn")
# learn.rewards = []



def plotStuff(title, xs,ys):
    print(f"\n{title}")
    fig = tpl.figure()
    fig.plot(xs, ys[1:], width=80, height=20)
    fig.show()


def smoothAndIndex(ys_orig, smooth):
    ys = [mean(ys_orig[i:i+smooth]) for i in range(0,len(ys_orig),smooth)][:-1]
    xs = list(range(len(ys)))
    return xs,ys



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--model_path", default="./latest_model.pt")
    parser.add_argument("--opt_path", default="./latest_model_opt.pt")
    parser.add_argument("--state_dim", type=int, default=5)
    parser.add_argument("--CEF_no", type=int, default=75)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--backlog_size", default=200, type=int)
    args = parser.parse_args()

    my_profiler = Profiler()

    if args.load_model:
        print("loading model...")
        policy_net = torch.load(args.model_path)
        # optimizer = torch.optim.SGD(policy_net.parameters(), lr = args.lr, momentum=0.9, nesterov=True)
        optimizer = torch.optim.Adam(policy_net.parameters(), lr = args.lr)
        optimizer.load_state_dict(torch.load(args.opt_path))
    else:
        print("not loading model...")
        # policy_net = PolicyNet(args.state_dim, 100, args.CEF_no, 2)
        # policy_net = PolicyNetUniform(args.CEF_no)
        policy_net = PolicyNetConstCategorical(args.state_dim, args.CEF_no)
        # policy_net = PolicyNetAttn(args.state_dim, args.CEF_no, 20)
        # optimizer = torch.optim.SGD(policy_net.parameters(), lr = args.lr, momentum=0.9, nesterov=True)
        optimizer = torch.optim.Adam(policy_net.parameters(), lr = args.lr)

    losses = []
    episodes={}

    best_loss = 10000.0

    with my_profiler.profile("save_policy"):
        print("Saving Initial policy...", end='')
        save_latest_policy(policy_net, optimizer, args.model_path, args.opt_path)
        print("done")

    while True:

        if len(episodes) > args.backlog_size:
            with my_profiler.profile("learn"):
                learn(episodes, policy_net, optimizer, losses, args.batch_size, args.backlog_size)
            with my_profiler.profile("save_policy"):
                print("Saving policy...", end='')
                save_latest_policy(policy_net, optimizer, args.model_path, args.opt_path)
                print("done")
        else:
            my_profiler.report()

            if len(losses) > 20:
                print(f"len(losses): {len(losses)}")
                totalLosses, actorLosses, criticLosses, entropy = list(zip(*[(loss['total'], loss['actor'], loss['critic'], loss['entropy']) for loss in losses]))
                delta = len(losses) // 20
                loss_ys = [mean(totalLosses[i:i+delta]) for i in range(0,len(losses),delta) if len(losses[i:i+delta]) > (delta-2)]
                actor_ys = [mean(actorLosses[i:i+delta]) for i in range(0,len(losses),delta) if len(losses[i:i+delta]) > (delta-2)]
                entropy_ys = [mean(entropy[i:i+delta]) for i in range(0,len(losses),delta) if len(losses[i:i+delta]) > (delta-2)]
                xs = list(range(len(loss_ys)-1))
                
                plotStuff("Total Loss Plot", xs, loss_ys)
                plotStuff("Actor Loss Plot", xs, actor_ys)

                if hasattr(policy_net, "critic"):
                    critic_ys = [mean(criticLosses[i:i+delta]) for i in range(0,len(losses),delta) if len(losses[i:i+delta]) > (delta-2)]
                    plotStuff("Critic Loss Plot", xs, critic_ys)

                plotStuff("Entropy Plot", xs, entropy_ys)

                if loss_ys[-1] < best_loss and len(losses) > 100:
                    save_latest_policy(policy_net, optimizer, args.model_path+".best", args.opt_path+".best")
                    best_loss = loss_ys[-1]

        newEpisodes = load_episodes(episodes)

        if len(load_episodes.historicalRewardInfo) > 20:
            with my_profiler.profile("PlottingHistoricalRewardInfo"):
                smoothness = len(load_episodes.historicalRewardInfo)//5
                xs,ys = smoothAndIndex(load_episodes.historicalRewardInfo, smooth=smoothness)
                if len(xs) > 1:
                    print(f"Smoothness: {smoothness}")
                    plotStuff("Proof Attempt Success Rate Plot", xs,ys)


        windowSize = 5*2078
        if not keepTraining(newEpisodes, windowSize=windowSize):
            print(f"Last {windowSize} episodes had no new solutions. Stopping training.")
            break
        





