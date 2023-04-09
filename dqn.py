import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math





EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 5000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



steps_done = 0
def select_action(state, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            qs = policy_net(state)
            return torch.distributions.Categorical(torch.softmax(qs, dim=1)).sample().view(1,1)
            # return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(args.CEF_no)]], device=device, dtype=torch.long)






def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)
    # IPython.embed()
    state_action_values = state_action_values.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()
    return loss.item()





class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)







class DQN(nn.Module):

    def __init__(self, inDim, midDim, outDim, numHidden):
        super(DQN, self).__init__()

        if numHidden < 1:
            raise Exception("For now, do at least 1 hidden layer!")

        self.input = nn.Linear(inDim, midDim)
        self.hidden = nn.ModuleList([nn.Linear(midDim, midDim) for _ in range(numHidden)])
        self.output = nn.Linear(midDim, outDim)
        

    def forward(self, states):

        x = F.relu(self.input(states))
        for layer in self.hidden:
            x = F.relu(layer(x))

        action_dist = self.output(x)
        return action_dist
