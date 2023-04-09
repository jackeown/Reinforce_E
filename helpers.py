import os, struct, select
import torch
from time import time, sleep

# Named Pipe Helpers ##########################################################


def sleepUntilHasData(pipe, timeout):
    print(" (sthd: ", end='')
    r = []
    t1 = time()
    while len(r) == 0 and (time() - t1) < timeout:
        print('.', end='')
        r, w, e, = select.select([pipe],[],[], 1)

    print(") ")
    return len(r) > 0
    

def read(pipe, n, timeout=300):
    print(f"Trying to read {n} bytes")
    t1 = time()
    bytez = b''
    while len(bytez) < n and (time() - t1) < timeout:
        if len(bytez) > 0:
            print(f"Read so far ({len(bytez)}/{n}): ", bytez)
        if sleepUntilHasData(pipe, timeout):
            newBytes = os.read(pipe, n-len(bytez))
            bytez += newBytes
            if len(newBytes) == 0 and len(bytez) == 0:
                sleep(1)
                # print("os.read returned b''")
                # break
    
    print(f"read {len(bytez)} bytes")
    if len(bytez) < n:
        return None
    
    return bytez

def write(pipe, bytez, timeout=30):
    num_written = 0
    t1 = time()
    while num_written < len(bytez) and time() - t1 < timeout:
        num_written += os.write(pipe, bytez[num_written:])
    
    if num_written == len(bytez):
        return True
    return False

def initPipe(pipePath, send=False, log=True):
    if log:
        print(f"Initializing Pipe ({pipePath})")

    mode = os.O_WRONLY if send else os.O_RDONLY

    retval = os.open(pipePath, mode)
    
    if log:
        print(f"Finished Initializing Pipe ({pipePath})")
    return retval


def normalizeState(state):
    # return torch.log(state + 2.0)
    # s = state / torch.tensor([1_000, 1_000, 2_000, 50, 50], dtype=torch.float)
    s = state / torch.tensor([40_000, 9_000, 1_300_000, 100, 200], dtype=torch.float)
    # s = state / torch.tensor([70_000, 16_000, 2_300_000, 150, 400], dtype=torch.float)
    
    # print(s)
    return s

def recvState(StatePipe, sync_num, CEF_no):
    stateSize = 4 + 3*8 + 2*4
    bytez = read(StatePipe, stateSize)
    if bytez is None:
        return None, False

    [sync_num_remote, ever_processed, processed, unprocessed, processed_weight, unprocessed_weight] = struct.unpack("=iqqqff", bytez)
    episode_begin = (sync_num_remote == 0 and sync_num > 0)
    assert (episode_begin or sync_num_remote == sync_num), f"{sync_num_remote} != {sync_num}"
    

    tensor = torch.tensor([ever_processed, processed, unprocessed, processed_weight, unprocessed_weight]).reshape(1, -1)
    
    # "Normalize" using log as per Geoff's suggestion.
    tensor = normalizeState(tensor)

    return tensor, episode_begin


def sendAction(action, pipe, sync_num):
    return write(pipe, struct.pack("=ii", sync_num, action.item()))
    


def recvReward(pipe, sync_num):
    rewardSize = 4 + 4
    bytez = read(pipe, rewardSize)
    if bytez is None:
        return None

    [sync_num_remote, reward] = struct.unpack("=if", bytez)
    assert sync_num_remote == sync_num, f"{sync_num_remote} != {sync_num}"

    assert reward in [0.0, 1.0]
    reward = 1.0 if reward == 1.0 else 0.0
    return torch.tensor(reward).reshape(1)



def sendProbId(pipe, id):
    write(pipe, struct.pack("=i", id))


def recvProbId(pipe):
    chars = read(pipe, 4)
    if chars is None:
        return None
        
    [id] = struct.unpack("=i", chars)
    return id




# Episode Helpers #############################################################

from collections import namedtuple
Episode = namedtuple('Episode', ['problem', 'states', 'actions', 'rewards', 'policy_net'])




# Generic Helpers #############################################################

def mean(l):
    return sum(l) / len(l)



