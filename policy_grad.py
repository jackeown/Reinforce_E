import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import IPython
import numpy as np
from collections import deque
import sys
from helpers import mean, normalizeState

class DummyProfiler:

    @staticmethod
    def profile(s):
        return DummyContextManager()

class DummyContextManager:

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, a,b,c):
        pass



def calculateReturnsAndAdvantageEstimate(policy, episode, GAMMA=0.999, LAMBDA=0.95, lunarLander=False):

    if len(episode.states) == 0:
        return np.array([]), np.array([])

    if lunarLander:
        states = torch.from_numpy(episode.states).to(torch.float)
    else:
        states = normalizeState(torch.from_numpy(episode.states).to(torch.float))

    actions = torch.from_numpy(episode.actions).to(torch.long)
    soft = torch.softmax(policy(states), dim=1)
    log_probs = torch.distributions.Categorical(soft).log_prob(actions).detach().numpy()
    values = policy.critic(states).reshape(-1).detach().numpy()
    reward = episode.rewards

    values_shifted = np.zeros_like(values)
    values_shifted[:-1] = values[1:]

    n = len(values)

    advantage = np.zeros(n + 1)
    returns = np.zeros(n + 1)

    not_last_timestep = 0
    for t in reversed(range(n)):
        returns[t] = reward[t] + GAMMA * returns[t + 1]
        td_err = reward[t] + (GAMMA * values_shifted[t] * not_last_timestep) - values[t]
        advantage[t] = td_err + (GAMMA * LAMBDA * advantage[t + 1] * not_last_timestep)
        not_last_timestep = 1

    biased_target = advantage[:n] + np.squeeze(values)
    noisy_target = returns[:n]
    
    how_much_bias = 0.85
    # value_target = biased_target
    value_target = how_much_bias*biased_target + (1-how_much_bias)*noisy_target

    return value_target, advantage[:n], log_probs, values
    # return noisy_target, (noisy_target - values), log_probs, values











def calculateReturns(policy_net, episode, off_policy, profiler, numpy=False, discount_factor=0.99):

    with profiler.profile("calculateReturns"):
        returns = deque()
        total = 0

        # returns = rewards when rewards are all 0
        if all(x == 0.0 for x in episode.rewards):
            return episode.rewards

        for s,a,r in reversed(list(zip(episode.states, episode.actions, episode.rewards))):

            importance = 1
            if off_policy:
                likelihood1 = torch.softmax(policy_net(s), dim=1)[0][a].detach().item()
                likelihood2 = torch.softmax(episode.policy_net(s), dim=1)[0][a].detach().item()
                try:
                    importance = (likelihood1 / likelihood2)
                except ZeroDivisionError:
                    importance = 1e-100
            
                if importance < 0.001 or importance > 99.9:
                    print(f"Ridiculous importance: {importance}")
                    return None

            total = importance * (r.item() + total * discount_factor)
            returns.appendleft(total)
        
        if numpy:
            returns = np.array(returns)
        else:
            returns = torch.tensor(returns)

        # Return normalization...
        if hasattr(policy_net, "critic"):
            return returns
        else:
            sys.exit()
            return (returns - returns.mean()) / (returns.std() + 1e-7)


def getGradNorm(policy_net, log=True, maxGradNorm=0.5):
    total_norm = 0.0
    for p in policy_net.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if log:
        print(f"Gradient Norm: {total_norm}")

    if maxGradNorm > 0:
        gradNorm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), maxGradNorm)
        if gradNorm > maxGradNorm and log:
            print(f"Gradient Clipping: (norm = {gradNorm})")

    return total_norm



def batch_generator(rollout_buffer, batch_size):
    random.shuffle(rollout_buffer)
    batch = []
    i = 0
    while i < len(rollout_buffer):
        batch.append(rollout_buffer[i])
        if len(batch) == batch_size:
            states,actions,returns,adv,log_probs,vals = list(zip(*batch))
            to_yield = [
                torch.stack(states).reshape(batch_size,-1),
                torch.stack(actions).reshape(batch_size),
                torch.stack(returns).reshape(batch_size),
                torch.stack(adv).reshape(batch_size),
                torch.stack(log_probs).reshape(batch_size),
                torch.stack(vals).reshape(batch_size),
            ]
            # print(to_yield)
            yield to_yield
            batch = []
        i += 1



def optimize_step_ppo(opt, policy_net, rollout_buffer, batch_size=128, critic_weight=0.1, entropy_weight=0.001, max_grad_norm=0.5, epochs=10):
    
    # old_log_probs = torch.distributions.Categorical(torch.softmax(states, dim=1)).log_prob(actions).detach()
    # old_values = policy_net.critic(states).detach()
    # rollout_buffer = list(zip(states, actions, value_target, advantages, old_log_probs, old_values))

    # batch_size = len(rollout_buffer)

    saved_info = []
    for epoch in range(epochs):
        for S, A, VAL_TARGET, ADV, old_log_probs, old_values in batch_generator(rollout_buffer, batch_size):
            # Get action log probabilities.
            soft = torch.softmax(policy_net(S),dim=1)
            dist = torch.distributions.Categorical(soft)
            log_probs = dist.log_prob(A)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_probs - old_log_probs)
            # print(log_probs.exp(), old_log_probs.exp())

            # clipped surrogate loss
            actor_loss_unclipped = ADV * ratio
            actor_loss_clipped = ADV * torch.clamp(ratio, 0.8, 1.2)
            actor_loss = -torch.min(actor_loss_unclipped, actor_loss_clipped).mean()


            values = policy_net.critic(S).reshape(-1)

            # Clipping values
            # max_val_change = values.mean() / 15
            max_val_change = 0.01
            values_pred = old_values + torch.clamp(values - old_values, -max_val_change, max_val_change)
            critic_loss = critic_weight * F.mse_loss(values_pred, VAL_TARGET)

            # critic_loss = critic_weight * F.mse_loss(values, VAL_TARGET)
            entropy_loss = -entropy_weight * dist.entropy().mean()

            loss = actor_loss + critic_loss + entropy_loss
            # loss = actor_loss + critic_loss
            
            opt.zero_grad()
            loss.backward()
            gradNorm = getGradNorm(policy_net, log=False, maxGradNorm=max_grad_norm)

            opt.step()

            saved_info.append([
                loss.item(),actor_loss.item(),critic_loss.item(),entropy_loss.item(), 
                gradNorm, ADV.mean().item()
            ])

    keys = ["total","actor","critic","entropy", "grad_norm", "advantage"]
    vals = zip(*saved_info)
    return {k:mean(v) for k,v in zip(keys,vals)}



def optimize_step(opt, policy_net, states, actions, returns, profiler=None, batch_size=64, critic_weight=0.1, entropy_weight=0.001, max_grad_norm=0.5):
    opt.zero_grad()

    if profiler is None:
        profiler = DummyProfiler
    
    with profiler.profile("forward pass"):
        dist = torch.distributions.Categorical(torch.softmax(policy_net(states), dim=1))
        entropy = dist.entropy()

        if hasattr(policy_net, "critic"):
            values = policy_net.critic(states).reshape(-1)
            advantage = returns - values

            critic_loss = F.mse_loss(values, returns, reduction="mean")
            # critic_loss = F.smooth_l1_loss(values, returns, reduction="mean")
            actor_loss = (-dist.log_prob(actions) * advantage.detach()).mean()
            entropy_loss = (-entropy).mean()

            critic_loss *= critic_weight
            entropy_loss *= entropy_weight

            loss = actor_loss + critic_loss + entropy_loss
        else:
            loss = -dist.log_prob(actions) * (returns)
            loss = loss.sum()

    with profiler.profile("backward pass"):
        loss.backward()
        gradNorm = getGradNorm(policy_net, log=False, maxGradNorm=max_grad_norm)
        opt.step()
    
    return {
        'total': loss.item(),
        'actor': actor_loss.item() if hasattr(policy_net, "critic") else loss.item(), 
        'critic': critic_loss.item() if hasattr(policy_net, "critic") else None, 
        'entropy': entropy_loss.item() if hasattr(policy_net, "critic") else entropy.item(),
        'grad_norm': gradNorm,
        'episode_length': len(actions),
        'solved': int(any(returns > 1e-6)),
        'advantage': advantage.mean().item()
    }
    




def select_action(policy_net, state):

    # if the number of given clause selections is crazy high,
    # we have little training data and it may be better to simply act randomly.
    # if state[0][0] > 2.0:
    #     return torch.randint(0,state.shape[1], (state.shape[0],1))

    out = policy_net(state)
    dist = torch.softmax(out, dim=1)
    return torch.distributions.Categorical(dist).sample().view(-1,1)




class Critic(nn.Module):

    def __init__(self, inDim, midDim, outDim):
        super().__init__()
        self.l1 = nn.Linear(inDim, midDim)
        self.l2 = nn.Linear(midDim, midDim)
        self.l3 = nn.Linear(midDim, outDim)

    def forward(self, x):
        a1 = F.relu(self.l1(x))
        a2 = F.relu(self.l2(a1))
        # return self.l3(a2)
        return torch.sigmoid(self.l3(a2))


class PolicyNetAttn(nn.Module):
    def __init__(self, stateDim, outDim, numDists):
        super().__init__()
        self.default = nn.Parameter(torch.randn(outDim))
        self.defaultWeight = nn.Parameter(torch.tensor(0.5))
        self.keys = nn.Parameter(torch.randn(numDists, stateDim))
        self.values = nn.Parameter(torch.randn(numDists, outDim))
        

    def forward(self, states):
        affinities = torch.softmax(states @ self.keys.T, 1)
        queryResult = (affinities.T * self.values).sum(dim=0).reshape(1,-1)
        weightedQueryResult = (1.0 - self.defaultWeight) * queryResult
        weightedDefault = self.defaultWeight * self.default
        return weightedQueryResult + weightedDefault


class PolicyNetUniform(nn.Module):
    def __init__(self, outDim):
        super().__init__()
        self.x = nn.Parameter(torch.ones(1,1))
        self.outDim = outDim
        
    def forward(self, states):
        return torch.ones(states.shape[0], self.outDim) * self.x


class PolicyNetVeryBad(nn.Module):
    """Confidently predicts class one always..."""
    def __init__(self, outDim):
        super().__init__()
        self.x = nn.Parameter(torch.ones(1,1))
        self.outDim = outDim
        
    def forward(self, states):
        stupid_one = (torch.sigmoid(self.x) / 1000) + 1
        out = torch.zeros(states.shape[0], self.outDim)
        out[:,0] = 100
        return out * stupid_one


class PolicyNetConstCategorical(nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.dist = nn.Parameter(torch.ones(outDim))
        self.critic = Critic(inDim, 100, 1)
        
    def forward(self, states):
        return self.dist.repeat(states.shape[0], 1)
    

class PolicyNetRoundRobin(nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.n = 0
        self.critic = Critic(inDim, 100, 1)
        self.outDim = outDim
        
    def forward(self, states):
        self.n += 1
        chosen = self.n % self.outDim
        return torch.tensor([100. if i == chosen else 0. for i in range(self.outDim)]).reshape(1,-1)

class PolicyNet(nn.Module):

    def __init__(self, inDim, midDim, outDim, numHidden):
        super().__init__()

        self.input = nn.Linear(inDim, midDim)
        self.normalizations = nn.ModuleList([nn.LayerNorm(midDim) for _ in range(numHidden)])
        self.hidden = nn.ModuleList([nn.Linear(midDim, midDim) for _ in range(numHidden)])
        self.output = nn.Linear(midDim, outDim)

        self.hook = nn.Linear(inDim, outDim)

        self.critic = Critic(inDim, 100, 1)
        

    def forward(self, states):

        first = self.input(states)
        x = first
        for layer, normalization in zip(self.hidden, self.normalizations):
            x = normalization(layer(x))
            x = F.relu(x)
            # x = F.leaky_relu(x, self.sqrt5) + first

        action_dist = self.output(x) + self.hook(states)
        return action_dist

