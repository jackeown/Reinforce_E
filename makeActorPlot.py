from e_caller import ECallerHistory
from helpers import normalizeState
import torch
from glob import glob
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
	print("Needs a positive int argument")
	sys.exit()
try:
	index = int(sys.argv[1])
except:
	print("Needs a positive int argument")
	sys.exit()

hist = ECallerHistory.load("nn_again4")
infos = sum(list(hist.history.values()), [])

successes = [info for info in infos if info['solved']]
big_succ = [i for i in successes if len(i['states']) > 2000]
succ = big_succ[index]

#model_paths = sorted(glob("model_histories/nn_third0_train/*"))
#model_paths = sorted(glob("model_histories/nn_short_blame_low_entropy0_train/*"))
#model_paths = sorted(glob("model_histories/nn_again0_train/*"))
model_paths = sorted(glob("model_histories/nn_again4_train/*"))

models = [torch.load(model_path) for model_path in model_paths]



def getPolicies(model, state):
	return [x.item() for x in torch.softmax(model(state), dim=1).reshape(-1)]

# First Plot is CEF preferences for the start state of succ as it changes over "models"
i = 0
#i = 1000
all_states = normalizeState(succ['states']).to(torch.float)
start_state = all_states[i].reshape(1,-1)
policies = [getPolicies(model, start_state) for model in models]
print(policies[-1])
time_series = list(zip(*policies))

xs = list(range(len(models)))
plt.stackplot(xs, *time_series)
plt.xlabel("Policy #")
plt.ylabel("CEF selection probabilities")
plt.xlim(0,len(xs))
plt.savefig(f"figures/actor_during_training_{succ['problem']}.png")


# The Second Plot is CEF preferences for the ith state of succ for the final model.
plt.figure()

model = models[-1]
policies = [getPolicies(model, state.reshape(1,-1)) for state in all_states]
time_series = list(zip(*policies))

xs = list(range(len(succ['states'])))
plt.stackplot(xs, *time_series)
plt.xlabel("Given Clause Selections")
plt.ylabel("CEF selection probabilities")
plt.xlim(0,2000)
plt.savefig(f"figures/actor_after_training_{succ['problem']}.png")
