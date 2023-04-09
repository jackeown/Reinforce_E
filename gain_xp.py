# This file communicates with Eprover using named pipes and generates
# experiences / rollouts / episodes using the latest model saved by `train_policy.py`
#
# (IT NO LONGER SAVES EPISODES. THIS IS NOW DONE WITH E_CALLER)

import argparse
import torch
from helpers import Episode, initPipe, recvState, sendAction, recvReward, recvProbId
from policy_grad import PolicyNet, select_action
from time import time, sleep
import numpy as np
from SerializationTesting import save_episode_numpy
import random

import IPython


def load_latest_policy(path):
    failed = True
    while failed:
        try:
            policy = torch.load(path)
            failed = False
        except:
            pass
        
    return policy

def save_episode(episode: Episode, numpy=False):
    print("Saving Episode")
    path = f"./episodes/{time()}.episode"
    if not numpy:
        torch.save(episode, path)
    else:
        save_episode_numpy(episode, path)





def rollout(policy_net, CEF_no, probIdPipe, initial_state=None):
    """Communicate with E until a proof has been found or a new proof attempt is started.

    + We detect the end of a proof attempt (successful or not) when recvState() gets a sync_num=0.
      We need to return the received state for the next call to rollout.

    + Therefore this function always returns (episode, next_rollout_init_state)
    + The initial_state is optional because the very first call to rollout won't have it.

    """

    sync_num = 0
    states, actions, rewards = [],[],[]

    probId = recvProbId(probIdPipe)
    while probId is None:
        probId = recvProbId(probIdPipe)
        print("Failed to recieve problem id from e_caller.py!")
        sleep(0.05)
    probName = f"MPT{probId:04}+1.p"
    print(f"probName is '{probName}'")

    while True:

        if initial_state is not None:
            state, episode_begin = initial_state, False
            initial_state = None
        else:
            print("Recieving State...", end="", flush=True)
            state, episode_begin = recvState(StatePipe, sync_num, CEF_no)
            print("Recieved State")

        if episode_begin:
            if 1.0 in rewards[:-1]:
                print("Episode should have ended earlier?!")
            return Episode(probName, states, actions, rewards, policy_net), state
        
        if state is None:
            print("STATE IS NONE")
            return Episode(probName, states, actions, rewards, policy_net), None

        try:
            action = torch.tensor(random.randint(0,CEF_no-1))
            # action = select_action(policy_net, state)
        except:
            print("# Failed to select action. IPython good sir?")
            IPython.embed()

        print("Sending Action...", end="", flush=True)
        sendAction(action, ActionPipe, sync_num)
        print(f"Sent Action: {action.item()}")
        print("Receiving Reward...", end='', flush=True)
        reward = recvReward(RewardPipe, sync_num)
        print("Received Reward")

        if reward is not None:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        else:
            print("REWARD IS NONE!!!")

        sync_num += 1




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./latest_model.pt")
    parser.add_argument("--CEF_no", type=int, default=20)
    parser.add_argument("--load_latest_interval", type=int, default=1)
    parser.add_argument("--num", type=int, default=1)
    args = parser.parse_args()

    ProbIdPipe = initPipe(f"/tmp/ProbIdPipe{args.num}", send=False)
    StatePipe = initPipe(f"/tmp/StatePipe{args.num}", send=False)
    ActionPipe = initPipe(f"/tmp/ActionPipe{args.num}", send=True)
    RewardPipe = initPipe(f"/tmp/RewardPipe{args.num}", send=False)

    episode_num = 0
    next_rollout_init_state = None

    print("Starting Loop")
    while True:

        if episode_num % args.load_latest_interval == 0:
            policy_net = load_latest_policy(args.model_path)

        print("starting rollout...")
        episode, next_rollout_init_state = rollout(policy_net, args.CEF_no, ProbIdPipe, next_rollout_init_state)

        if len(episode.rewards) == 0:
            print("Proof not found! (No rewards?!)")
            exit()
        elif episode.rewards[-1] > 0:
            print("Proof Found!")
            # save_episode(episode)
        else:
            print("Proof not found!")
            max_blame = 200000
            rewards = episode.rewards[:max_blame]
            rewards[-1] = torch.tensor(-1.0).reshape(1)

            episode = Episode(
                episode.problem, 
                episode.states[:max_blame], 
                episode.actions[:max_blame], 
                rewards, 
                episode.policy_net
            )
            # save_episode(episode)

        episode_num += 1
