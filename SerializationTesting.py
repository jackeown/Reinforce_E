
from glob import glob
from my_profiler import Profiler
from helpers import Episode
import torch
import pickle
import multiprocessing
from functools import partial
from multiprocessing import Lock
import numpy as np



def save_episode_numpy(ep, path):
    states = [x.detach().numpy() for x in ep.states]
    actions = [x.detach().numpy() for x in ep.actions]
    rewards = [x.detach().numpy() for x in ep.rewards]

    to_dump = Episode(ep.problem, states, actions, rewards, None)
    with open(path, "wb") as f:
        f.write(pickle.dumps(to_dump, protocol=0))
        

def npToTorch(ep):
    states = [torch.from_numpy(x) for x in ep.states]
    actions = [torch.from_numpy(x) for x in ep.actions]
    rewards = [torch.from_numpy(x) for x in ep.rewards]
    return Episode(ep.problem, states, actions, rewards, None)

def npToTorchFasterMaybe(ep):
    # It's not faster yet...

    ranges = []
    cursor = 0
    for x in ep.states:
        new_cursor = cursor + x.shape[0]
        ranges.append([cursor, new_cursor])
        cursor = new_cursor

    states = torch.from_numpy(np.concatenate(ep.states, axis=0))
    actions = torch.from_numpy(np.concatenate(ep.actions, axis=0))
    rewards = torch.from_numpy(np.concatenate(ep.rewards, axis=0))

    states, actions, rewards = list(zip(*[(states[begin:end], actions[begin:end], rewards[begin:end]) for begin, end in ranges]))

    return Episode(ep.problem, states, actions, rewards, None)

def load_episode_numpy(path):
    with open(path, 'rb') as f:
        ep = pickle.loads(f.read())
    return npToTorch(ep)

def load_episode_numpy_locked(lock, path):
    """Doesn't currently convert to torch for you as that can't be done in parallel apparently..."""
    with open(path, 'rb') as f:
        ep = f.read()

    ep = pickle.loads(ep)
    return ep


if __name__ == "__main__":

    from train_policy import load_episode

    profiler = Profiler()
    episodePaths = glob("episodes/*.episode")
    episodes = {}



    print("Normal Loading...")
    with profiler.profile("For loop normal loading..."):
        for ep in episodePaths:
            episodes[ep] = load_episode(ep)

    # print("Saving back with numpy...")
    # with profiler.profile("Saving numpy..."):
    #     for path,ep in episodes.items():
    #         save_episode_numpy(ep, path+".np")

    print("Loading with numpy...")
    with profiler.profile("For loop numpy loading..."):
        for path in episodePaths:
            episodes[path] = load_episode_numpy(path+".np")


    print("Parallel loading with numpy...")
    with profiler.profile("Parallel numpy loading..."):
        with multiprocessing.Manager() as manager:
            l = manager.Lock()
            f = partial(load_episode_numpy_locked, l)

            with multiprocessing.Pool(80) as p:
                with profiler.profile("parallel part"):
                    eps = p.map(f, [p+".np" for p in episodePaths])
                with profiler.profile("conversion to torch"):
                    eps = [npToTorchFasterMaybe(ep) for ep in eps]
                # eps = [npToTorch(ep) for ep in eps]


    profiler.report()