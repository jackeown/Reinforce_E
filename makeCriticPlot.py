from e_caller import ECallerHistory
from helpers import normalizeState
import torch
import matplotlib.pyplot as plt

def getSuccessesAndFailures(run):
	hist = ECallerHistory.load(run)
	infos = sum(list(hist.history.values()), [])
	successes = [info for info in infos if info['solved'] and len(info['states'])>0]
	failures = [info for info in infos if not info['solved'] and len(info['states'])>0]
	return successes,failures

def criticValsFromInfos(critic, infos):
	vals = []
	for info in infos:
		state = normalizeState(info['states']).to(torch.float)[0].reshape(1,-1)
		vals.append(critic(state).item())
	return vals


mean = lambda l: sum(l) / len(l)

nbins = 20
height = 200

stuff = [getSuccessesAndFailures(run) for run in ["nn_second0","nn_second1","nn_second2","nn_second3","nn_second4"]]
successes, failures = list(zip(*stuff))

models = [torch.load(f"models/nn_second{i}.pt") for i in range(5)]
successVals = [criticValsFromInfos(model.critic, s) for s,model in zip(successes,models)]
failureVals = [criticValsFromInfos(model.critic, f) for f,model in zip(failures,models)]

print([mean(x) for x in successVals])
print([mean(x) for x in failureVals])

successVals, failureVals = sum(successVals,[]), sum(failureVals,[])

plt.hist(failureVals, bins=nbins, label="failures", alpha=0.5, color="red", hatch="\\\\")
plt.hist(successVals, bins=nbins, label="successes", alpha=0.5, color="green", hatch="//")
plt.legend()
plt.ylim(0,height)
#plt.title('Critic Output Histogram For Initial States')
plt.xlabel("Critic Output")
plt.ylabel("# Problems with Critic Output Approximately X")
plt.savefig("figures/critic_plot.png")




#plt.figure()
#print("Success average: ", mean(successVals))
#plt.hist(successVals, bins=nbins)

#plt.ylim(0,height)
#plt.title('State Value Histogram (successes)')
#plt.xlabel("Critic Output")
#plt.ylabel("Histogram Bin Count")
#plt.savefig("critic_plot_success.png")

#plt.figure()
#print("Failure average: ", mean(failureVals))
#plt.hist(failureVals, bins=nbins)

#plt.ylim(0,height)
#plt.title('State Value Histogram (failures)')
#plt.xlabel("Critic Output")
#plt.ylabel("Histogram Bin Count")
#plt.savefig("critic_plot_failure.png")











