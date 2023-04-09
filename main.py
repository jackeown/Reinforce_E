########################################
# Trying to combine everything so that #
# I don't have any disk overhead.      #
##################################################################################################
# Function I want:                                                                               #
# InvokeE(policy, problem, cpu_limit, cefs_file) -> (states, actions, rewards) tensors            #
#                                                                                                #
# InvokeE duties:                                                                                #
# 1.) Start E subprocess.                                                                        #
# 2.) Communicate with E back and forth receiving states and sending actions as fast as possible.#
# 3.) Recognize proof attempt end (via 1 reward or closing of named pipe?)                       #
# 4.) Extract rewards from E stdout.                                                             #
##################################################################################################

from collections import defaultdict
from email import message
import os, re, struct, sys
import argparse
import subprocess, threading
import itertools, functools
import random
import multiprocessing as mp
import subprocess
from time import time, sleep
from glob import glob
import resource
from typing import Iterable, List

import torch
import numpy as np

print_dumb = print
from rich import print
from rich.progress import track
import rich_dashboard

import IPython

from my_profiler import Profiler


from helpers import Episode, mean, initPipe, normalizeState
from e_caller import ECallerHistory, extractStatus, extractProcessedCount, extractOverhead, getPositiveClauses, getStates, getActions, getRewards, getConfigName, getConfig, getGivenClauses
from policy_grad import PolicyNet, PolicyNetConstCategorical, PolicyNetUniform, PolicyNetAttn, optimize_step_ppo, select_action, optimize_step, calculateReturns, calculateReturnsAndAdvantageEstimate, DummyProfiler


dummyProfiler = DummyProfiler()

def clonePolicy(module):
    copy,_ = getPolicy() # get a new instance
    copy.load_state_dict(module.state_dict()) # copy weights and stuff
    return copy

def grouper(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def createInfoFromE(stdout, stderr, prob, t1, t2):

    info = {}

    info["status"] = extractStatus(stdout)
    if info['status'] == "No SZS Status":
        print(f"No szs status? sus: {prob}")
        print_dumb("stdout: ",stdout[:1000] + "..."*100 + stdout[-1000:])
        print_dumb("stderr: ",stderr)


    if "Segmentation fault" in stderr:
        print(f"SEGFAULT for {prob}")


    info["solved"] = ("Unsatisfiable" in info["status"]) or ("Theorem" in info["status"])

    info['states'] = np.array(getStates(stdout))
    info['actions'] = np.array(getActions(stdout))
    # print("Getting rewards...")
    info['rewards'] = np.array(getRewards(stdout, len(info['states'])))
    # print("Finished getting rewards")

    # info['configName'] = getConfigName(stderr, yell=False)
    # info['cefs'] = getConfig(stdout, stderr, info['configName'], yell=False)



    try:
        info['args'] = args
    except NameError:
        info['args'] = None


    info["time"] = t2 - t1
    info["timestamp"] = t1
    info["processed_count"] = len(getGivenClauses(stdout))
    info["processed_count_reported"] = extractProcessedCount(stdout) if info["solved"] else float('inf')
    info["problem"] = prob
    info["probId"] = int(prob[-8:-4])
    info["statePipeTime"], info["actionPipeTime"], info["rewardPipeTime"], info["prepTime"] = extractOverhead(stdout)


    # lines = stdout.split("\n")
    # info['posClauses'] = getPositiveClauses(lines)
    # info['givenClauses'] = getGivenClauses(stdout)
    
    limit = 5_000
    # limit = 100_000_000
    info['stdout'] = "\n".join(stdout.split("\n")[-limit:])
    info['stderr'] = "\n".join(stderr.split("\n")[-limit:])

    return info




###############################################################################
#                                                                             #
#                              PIPE HELPERS                                   #
#                                                                             #
###############################################################################


def recvState(StatePipe, sync_num, CEF_no):
    stateSize = 4 + 3*8 + 2*4
    bytez = b''
    while len(bytez) < stateSize:
        new = os.read(StatePipe, stateSize-len(bytez))
        if len(new) == 0:
            return None
        bytez += new

    [sync_num_remote, ever_processed, processed, unprocessed, processed_weight, unprocessed_weight] = struct.unpack("=iqqqff", bytez)
    assert (sync_num_remote == sync_num), f"{sync_num_remote} != {sync_num}"


    state = torch.tensor([ever_processed, processed, unprocessed, processed_weight, unprocessed_weight]).reshape(1, -1)
    state = normalizeState(state)

    return state


def sendAction(action, pipe, sync_num):
    bytes_written = 0
    to_write = struct.pack("=ii", sync_num, action.item())
    while bytes_written < len(to_write):
        bytes_written += os.write(pipe, to_write[bytes_written:])


def recvReward(pipe, sync_num):
    rewardSize = 4 + 4
    bytez = b''
    while len(bytez) < rewardSize:
        new = os.read(pipe, rewardSize - len(bytez))
        if len(new) == 0:
            return None
        bytez += new

    [sync_num_remote, reward] = struct.unpack("=if", bytez)
    assert sync_num_remote == sync_num, f"{sync_num_remote} != {sync_num}"

    assert reward in [0.0, 1.0]
    reward = 1.0 if reward == 1.0 else 0.0
    return torch.tensor(reward).reshape(1)




###############################################################################
#                                                                             #
#                             END PIPE HELPERS                                #
#                                                                             #
###############################################################################



def communicateWithE(policy, workerId):
    sync_num = 0

    StatePipe = initPipe(f"/tmp/StatePipe{workerId}", send=False, log=False)
    ActionPipe = initPipe(f"/tmp/ActionPipe{workerId}", send=True, log=False)
    RewardPipe = initPipe(f"/tmp/RewardPipe{workerId}", send=False, log=False)

    while True:
        try:
            state = recvState(StatePipe, sync_num, None)

            if state is None:
                return

            action = select_action(policy, state)
            sendAction(action, ActionPipe, sync_num)
            recvReward(RewardPipe, sync_num)

            sync_num += 1

        except OSError as e:
            print("OSError. Probably pipe closed")
            break


# For reading E's stdout/stderr while we interact with E in another thread
def read_output(process, result):
    stdout, stderr = [], []

    def gather_output(pipe):
        for line in iter(pipe.readline, b''):
            yield line

    for line in gather_output(process.stdout):
        stdout.append(line.decode())
    for line in gather_output(process.stderr):
        stderr.append(line.decode())
    process.wait()

    result.append(("".join(stdout), "".join(stderr)))



def makeEnv(workerId):
    env = os.environ.copy()
    env["E_RL_STATEPIPE_PATH"] = f"/tmp/StatePipe{workerId}"
    env["E_RL_ACTIONPIPE_PATH"] = f"/tmp/ActionPipe{workerId}"
    env["E_RL_REWARDPIPE_PATH"] = f"/tmp/RewardPipe{workerId}"

    if not os.path.exists(env['E_RL_STATEPIPE_PATH']):
        os.mkfifo(env['E_RL_STATEPIPE_PATH'])
    if not os.path.exists(env['E_RL_ACTIONPIPE_PATH']):
        os.mkfifo(env['E_RL_ACTIONPIPE_PATH'])
    if not os.path.exists(env['E_RL_REWARDPIPE_PATH']):
        os.mkfifo(env['E_RL_REWARDPIPE_PATH'])

    return env



def runE(policy, eproverPath, CEF_STRING, problemPath, soft_cpu_limit=1, cpu_limit=5, auto=False, create_info=True, verbose=False):
    problemName = os.path.split(problemPath)[1]
    workerId = random.randint(0,1_000_000_000)

    NORMAL_FLAGS = f"-l1 --proof-object --print-statistics --training-examples=3 --soft-cpu-limit={soft_cpu_limit} --cpu-limit={cpu_limit}"
    # MAGIC_FLAGS = "--simul-paramod --forward-context-sr --strong-destructive-er --destructive-er-aggressive --destructive-er -F1 -WSelectComplexExceptUniqMaxHorn -tKBO6 -winvfreqrank -c1 -Ginvfreq --strong-rw-inst"
    MAGIC_FLAGS = "--simul-paramod --forward-context-sr --strong-destructive-er --destructive-er-aggressive --destructive-er --presat-simplify -F1 -WSelectComplexExceptUniqMaxHorn -tKBO6 -winvfreqrank -c1 -Ginvfreq --strong-rw-inst"

    if auto:
        command_args = [
            eproverPath, *NORMAL_FLAGS.split(), "--auto", problemPath
        ]
    else:
        command_args = [
            eproverPath, *NORMAL_FLAGS.split(), *MAGIC_FLAGS.split(), "-H", f"{CEF_STRING}", problemPath
        ]


    if verbose and not create_info:
        p = subprocess.Popen(command_args, env=makeEnv(workerId))
        if policy is not None:
            communicateWithE(policy, workerId)
        p.wait()
        return

    # Run E itself with the args specified above.
    t1 = time()
    p = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=makeEnv(workerId))

    # Thread needed to read E's stdout/stderr incrementally
    result = []
    thread = threading.Thread(target=read_output, args=(p,result))
    thread.start()

    # Interact with E via named pipes using "policy" to map states to actions.
    if policy is not None:
        communicateWithE(policy, workerId)

    # Finish reading E's stdout/stderr
    thread.join()
    stdout,stderr = result[0]
    t2 = time()

    if verbose:
        print_dumb("About to print stdout/stderr...")
        print_dumb(stdout,stderr)

    # Extract important info from E's stdout/stderr
    if create_info:
        return createInfoFromE(stdout,stderr, problemName, t1, t2)
    else:
        return stdout,stderr





#################### Alternative Environment for testing RL #############################


def createInfoFromLunarLander(states, actions, rewards):
    info = {}
    info['solved'] = True
    info['problem'] = "LunarLander"
    info['states'] = np.array(states)
    info['actions'] = np.array(actions)
    info['rewards'] = np.array(rewards) / 200
    info['stdout'] = ''
    info['stderr'] = ''
    return info


def runLunarLander(policy, problem):
    import gymnasium as gym
    env = gym.make("LunarLander-v2")
    observation, info = env.reset(seed=random.randint(0,1_000_000))
    states, actions, rewards = [],[],[]
    for _ in range(10000):
        sleep(0.0001)
        states.append(observation)
        torch_obs = torch.from_numpy(observation).to(torch.float).reshape([1,-1])
        action = select_action(policy, torch_obs).item()
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            env.close()
            return createInfoFromLunarLander(states, actions, rewards)

#######################################################################################





def getPolicy(log=False):

    print_if_log = lambda x: print(x) if log else None

    if args.policy_type == "none" or args.auto:
        print_if_log("Model is None")
        return None, None
    
    if args.load_model:
        print_if_log("loading model...")
        policy = torch.load(args.model_path)
        if not args.test:
            opt = torch.optim.RMSprop(policy.parameters(), lr = args.lr)
            # opt = torch.optim.Adam(policy.parameters(), lr = args.lr)
            opt.load_state_dict(torch.load(args.opt_path))
        else:
            opt = None
    else:
        print_if_log("not loading model...")

        if args.policy_type == "constcat":
            policy = PolicyNetConstCategorical(args.state_dim, args.CEF_no)
        elif args.policy_type == "nn":
            policy = PolicyNet(args.state_dim, args.n_units, args.CEF_no, args.n_layers)
        elif args.policy_type == "uniform":
            policy = PolicyNetUniform(args.CEF_no)
        elif args.policy_type == "attn":
            policy = PolicyNetAttn(args.state_dim, args.CEF_no, args.n_units)
        else:
            print(f"No Policy Being Used!!!: args.policy_type = {args.policy_type}")

        opt = torch.optim.RMSprop(policy.parameters(), lr = args.lr)
        # opt = torch.optim.Adam(policy.parameters(), lr = args.lr)

    return policy, opt



def sendInfoAndEpisode(proc, info_queue, episode_queue, message_queue, policy):
    info = proc.get()
    info_queue.put(info)
    ep = Episode(info['problem'], info['states'], info['actions'], info['rewards'], policy)


    if len(ep.states) == 0:
        message_queue.put("len(ep.states) == 0!")


    if len(set([len(ep.states), len(ep.actions), len(ep.rewards)])) != 1:
        message = f"State / Action / Reward sizes Differ!: ({len(ep.states)},{len(ep.actions)},{len(ep.rewards)})"
        print(message)
        message_queue.put(message)

        min_len = min(len(ep.states), len(ep.actions), len(ep.rewards))
        ep = Episode(ep.problem, ep.states[:min_len], ep.actions[:min_len], ep.rewards[:min_len], ep.policy_net)


    # returns = calculateReturns(policy, ep, False, dummyProfiler,
    #     numpy=True, discount_factor=args.discount_factor)

    returnsAdvProbsAndVals = calculateReturnsAndAdvantageEstimate(policy, ep, GAMMA=args.discount_factor, LAMBDA=args.LAMBDA, lunarLander=args.lunar_lander)
    modified_ep = Episode(ep.problem, ep.states, ep.actions, returnsAdvProbsAndVals, policy)
    modified_ep = applyMaxBlame(modified_ep, args.max_blame)

    episode_queue.put((info['solved'], modified_ep))

    # probName and whether or not the problem was solved in presaturation interreduction.
    return info['problem'], len(ep.states) == 0 and info['solved']



def gather_episodes_process(policy_queue, problems, episode_queue, info_queue, message_queue, presat_info_queue, profiler_queue, stop_event):
    policy = policy_queue.get()
    if args.lunar_lander:
        runner = functools.partial(runLunarLander, policy)
    else:
        runner = functools.partial(runE, policy, args.eprover_path, CEF_STRING, soft_cpu_limit=args.soft_cpu_limit, cpu_limit=args.cpu_limit)

    random.seed(args.seed)

    i = 0
    profiler = Profiler()
    while not stop_event.value:

        processes = []
        sentCount = 0
        random.shuffle(problems)
        with mp.Pool(args.num_workers) as p:

            probsSolvedPresat = set()
            for prob in problems:
                result = p.apply_async(runner, (prob,))
                processes.append(result)


                with profiler.profile("<g> waiting for learner (rate limiting)"):
                    while episode_queue.qsize() > 5:
                        message_queue.put("Sleeping for 10 seconds. Waiting for learner...")
                        sleep(10)

                while len(processes[sentCount:]) >= args.batch_size*2:

                    # Get Latest Policy
                    with profiler.profile("<g> receiving latest policy"):
                        while policy_queue.qsize():
                            policy = policy_queue.get()
                        if args.lunar_lander:
                            runner = functools.partial(runLunarLander,policy)
                        else:
                            runner = functools.partial(runE, policy, args.eprover_path, CEF_STRING, soft_cpu_limit=args.soft_cpu_limit, cpu_limit=args.cpu_limit)

                    # Send as many unsent episodes as are ready.
                    with profiler.profile("<g> send done but unsent episodes"):
                        readiness = ",".join([str(1 if proc.ready() else 0) for proc in processes[sentCount:]])
                        message_queue.put(f"{len(processes)} {sentCount} ({readiness})")
                        for proc in processes[sentCount:]:
                            if proc.ready():
                                with profiler.profile("<g> sendInfoAndEpisode"):
                                    probName, solved_in_presat = sendInfoAndEpisode(proc, info_queue, episode_queue, message_queue, policy)
                                if probName not in probsSolvedPresat and solved_in_presat:
                                    probsSolvedPresat.add(probName)
                                processes[sentCount] = None
                                sentCount += 1
                            else:
                                sleep(0.1)
                                break # important to quit so that sentCount truly represents a cutoff point in processes.

                    profiler_queue.put(profiler.copy())
                    profiler.reset()

            while sentCount < len(processes):
                with profiler.profile("<g> sendInfoAndEpisode"):
                    probName, solved_in_presat = sendInfoAndEpisode(processes[sentCount], info_queue, episode_queue,message_queue,  policy)
                if probName not in probsSolvedPresat and solved_in_presat:
                    probsSolvedPresat.add(probName)
                sentCount += 1
        
        profiler_queue.put(profiler.copy())
        profiler.reset()
        presat_info_queue.put(probsSolvedPresat)
        message_queue.put(f"Finished the {i}th proof attempt of all Problems. On to the {i+1}th...")

        i += 1


def keepTraining(everSolved, patience=5*2078, keep_training_queue=None):

    if len(everSolved) > len(keepTraining.prevEverSolved):
        keepTraining.attemptsSinceNewSolution = 0
        keepTraining.prevEverSolved = set(everSolved)
        return True
    else:
        keepTraining.attemptsSinceNewSolution += 1
        keep_training_queue.put((len(keepTraining.prevEverSolved), keepTraining.attemptsSinceNewSolution, patience))
        return keepTraining.attemptsSinceNewSolution < patience

keepTraining.attemptsSinceNewSolution = 0
keepTraining.prevEverSolved = set()



def applyMaxBlame(ep, max_blame):

    if len(ep.rewards) == 0:
        return ep

    if isinstance(ep.rewards, tuple):
        if any(x != 0 for x in ep.rewards[0]):
            return ep
    elif any(x != 0 for x in ep.rewards):
        return ep

    return Episode(ep.problem, ep.states[:max_blame], ep.actions[:max_blame], ep.rewards[:max_blame], ep.policy_net)


def train_policy_process(policy, opt, episode_queue, policy_queue, info_queue, keep_training_queue, message_queue, profiler_queue, stop_event):

    policy_queue.put(clonePolicy(policy))

    batch = []
    model_iteration = 0
    profiler = Profiler()
    i = 0
    while keepTraining(train_policy_process.everSolved, args.train_patience, keep_training_queue):
        i += 1
        message_queue.put(f"train main loop {i}")

        with profiler.profile("<t> entire while body"):

            with profiler.profile("<t> Waiting for episodes"):
                while episode_queue.qsize() == 0:
                    message_queue.put("trainer waiting for episodes")
                    sleep(10)

            solved, ep = episode_queue.get()
            if solved and ep.problem not in train_policy_process.everSolved:
                message_queue.put(f"[cyan]{ep.problem}[/cyan] solved for the first time!")
                train_policy_process.everSolved.add(ep.problem)


            isNumeric = lambda x: np.issubdtype(x.dtype,np.number)
            allNumeric = all(isNumeric(x) for x in [ep.states, ep.actions, ep.rewards[0]])
            if not allNumeric:
                message_queue.put("Non numeric states, actions, or rewards encountered!")
                message_queue.put(",".join([str(x.dtype) for x in [ep.states, ep.actions, ep.rewards]]))


            if len(ep.states) > 0 and allNumeric:
                with profiler.profile("<t> Numpy -> torch"):
                    states = torch.from_numpy(ep.states).to(torch.float)
                    actions = torch.from_numpy(ep.actions).to(torch.long)
                    returns = torch.from_numpy(ep.rewards[0]).to(torch.float) # ep.rewards are actually (returns,advantages,log_probs,values)...see episode_queue.put
                    advantages = torch.from_numpy(ep.rewards[1]).to(torch.float)
                    log_probs = torch.from_numpy(ep.rewards[2]).to(torch.float)
                    values = torch.from_numpy(ep.rewards[3]).to(torch.float)

                with profiler.profile("<t> Normalizing State"):
                    if not args.lunar_lander:
                        states = normalizeState(states)
                batch.append([states, actions, returns, advantages, log_probs, values])

            if len(batch) >= args.batch_size:

                with profiler.profile("<t> Optimize_step"):
                    unzipped = [torch.cat(X, dim=0) for X in zip(*batch)]
                    rollout_buffer = list(zip(*unzipped))
                    info_queue.put((
                        opt, 
                        clonePolicy(policy),
                        optimize_step_ppo(opt, policy, rollout_buffer, args.ppo_batch_size, args.critic_weight, args.entropy_weight, args.max_grad_norm, args.epochs)
                        # optimize_step(opt, policy, states, actions, returns, None, 1, args.critic_weight, args.entropy_weight, args.max_grad_norm)
                    ))

                model_iteration += 1
                model_history_dir = f"model_histories/{args.run}"
                os.makedirs(model_history_dir, exist_ok=True)
                torch.save(policy, f"{model_history_dir}/{model_iteration:05d}.pt")

                policy_queue.put(clonePolicy(policy))
                batch = []
        
        profiler_queue.put(profiler.copy())
        profiler.reset()

    stop_event.value = True
train_policy_process.everSolved = set()



def logTraining(loss):

    print(loss)

    if logTraining.i < 2:
        for mode in loss:
            logTraining.runningLoss[mode].append(loss[mode])

    else:
        for mode in loss:
            if isinstance(logTraining.runningLoss[mode], Iterable):
                logTraining.runningLoss[mode] = mean(logTraining.runningLoss[mode])
            else:
                logTraining.runningLoss[mode] = 0.8*logTraining.runningLoss[mode] + 0.2*loss[mode]

        dashboard.updateLoss(loss, logTraining.runningLoss)

    logTraining.i += 1

logTraining.runningLoss = defaultdict(list)
logTraining.i = 0



def TrainPolicy(problems, args):
    policy, opt = getPolicy(log=True)
    history = ECallerHistory()

    # Primary Stuff (necessary for learning to make sense)
    episode_queue = mp.Queue()
    policy_queue = mp.Queue()
    stop_event = mp.Value('b', False)

    # Stuff used only for dashboard
    keep_training_queue = mp.Queue()
    message_queue = mp.Queue()
    gather_info_queue = mp.Queue()
    train_info_queue = mp.Queue()
    presat_info_queue = mp.Queue()
    profiler_queue = mp.Queue()

    # Start the process that calls E
    gatherProc = mp.Process(
        target = gather_episodes_process,
        args = (policy_queue, problems, episode_queue, gather_info_queue, message_queue, presat_info_queue, profiler_queue, stop_event)
    )
    print("Starting gatherer...")
    gatherProc.start()


    # Start the process that trains the model
    trainProc = mp.Process(
        target = train_policy_process,
        args = (policy, opt, episode_queue, policy_queue, train_info_queue, keep_training_queue, message_queue, profiler_queue, stop_event)
    )
    print("Starting trainer...")
    trainProc.start()


    # Update Dashboard during training.
    # Also save policy and history occasionally.
    while not stop_event.value:
        if gather_info_queue.qsize() > 0:
            info = gather_info_queue.get()
            history.addInfo(info)
            dashboard.updateRewardGraph(history)
            if not args.lunar_lander:
                dashboard.updateProofAttemptSuccessRateGraph(history)

        if presat_info_queue.qsize() > 0:
            dashboard.updatePresatInfo(presat_info_queue.get())
        while train_info_queue.qsize() > 0:
            worked = False
            try:
                opt, policy, loss = train_info_queue.get()
                worked = True
            except Exception as e:
                message_queue.put("Failed to train_info_queue.get()...trying again...")
            if worked:
                logTraining(loss)

        if keep_training_queue.qsize() > 0:
            probsEverSolved, attemptsSinceSolution, patience = keep_training_queue.get()
            dashboard.updateProbsEverSolved(probsEverSolved, attemptsSinceSolution, patience)
        if message_queue.qsize() > 0:
            dashboard.addMessage(message_queue.get())

        dashboard.updateQueueInfo({
            "Episode Queue: ": episode_queue.qsize(),
            "Policy Queue: ": policy_queue.qsize(),
            "Gather_Info Queue: ": gather_info_queue.qsize(),
            "Train Info Queue: ": train_info_queue.qsize(),
            "KeepTraining Queue: ": keep_training_queue.qsize(),
            "Message Queue: ": message_queue.qsize(),
            "Presat info Queue: ": presat_info_queue.qsize(),
            "Profiler Queue: ": profiler_queue.qsize()
        })

        while profiler_queue.qsize() > 0:
            dashboard.updateProfiler(profiler_queue.get())

        qs = [episode_queue, policy_queue, gather_info_queue, train_info_queue, keep_training_queue]
        max_queue_size = max(x.qsize() for x in qs)
        if max_queue_size < 5 and time() - TrainPolicy.lastSaved > 60:
            gatherState = "running" if gatherProc.is_alive() else "dead"
            trainState = "running" if trainProc.is_alive() else "dead"
            dashboard.addMessage(f"Saving Policy and History... (gatherer {gatherState}, trainer {trainState})")

            dead = {name for name,proc in zip(['gather','train'],[gatherProc, trainProc]) if not proc.is_alive()}
            
            history.save(f"{args.run}_train", 1, eager=True if dead else False)
            torch.save(policy, args.model_path)
            torch.save(opt.state_dict(), args.opt_path)
            TrainPolicy.lastSaved = time()
            
            if dead:
                print(f"The following processes died: {dead}")
                sys.exit()



        dashboard.render()

    gatherProc.join()
    trainProc.join()

    history.save(f"{args.run}_train", 1, eager=True)
    torch.save(policy, args.model_path)
    torch.save(opt.state_dict(), args.opt_path)

    return history

TrainPolicy.lastSaved = 0






def EvaluatePolicy(policy, problems, args):
    history = ECallerHistory()

    if args.lunar_lander:
        runner = functools.partial(runLunarLander, policy)
    else:
        runner = functools.partial(runE, policy, args.eprover_path, CEF_STRING, auto=args.auto, soft_cpu_limit=args.soft_cpu_limit, cpu_limit=args.cpu_limit)



    print("Evaluating...")
    with mp.Pool(args.num_workers) as p:
        for info in p.imap(runner, problems):
            print('+' if info['solved'] else '-', end='')
            history.addInfo(info)

    print(f"Number of Problems Solved: {len(history.probsSolved())}")
    return history








if __name__ == "__main__":

    torch.set_num_interop_threads(8)
    torch.set_num_threads(8)


    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))

    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default=os.path.expanduser("~/Desktop/ATP/GCS/MPTPTP2078/Bushy/Problems/"), help="path to where problems are stored")
    parser.add_argument("--run", default="test")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--lunar_lander", action="store_true")
    parser.add_argument("--auto", action="store_true")

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=4, help="number of episodes before training")
    parser.add_argument("--ppo_batch_size", type=int, default=128, help="Batch size for PPO updates")
    parser.add_argument("--n_units", type=int, default=100, help="Number of units per hidden layer in the policy")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of hidden NN layers in the policy")
    parser.add_argument("--discount_factor", type=float, default=0.998, help="discount factor for RL")
    parser.add_argument("--LAMBDA", type=float, default=0.95, help="PPO discount for interpolating between full returns and TD estimate")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for each PPO training phase")
    parser.add_argument("--critic_weight", type=float, default=0.4)
    parser.add_argument("--entropy_weight", type=float, default=4e-5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--max_blame", type=int, default=20_000, help="Maximum number of given clause selections to punish for a failed proof attempt...")
    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--model_path", default="latest_model.pt")
    parser.add_argument("--opt_path", default="latest_model_opt.pt")
    parser.add_argument("--policy_type", default="nn", choices=["nn", "constcat", "none", "uniform", "attn"])
    
    parser.add_argument("--state_dim", type=int, default=5)
    parser.add_argument("--CEF_no", type=int, default=20, help="Number of cefs in cef_file")
    parser.add_argument("--cef_file", default="cefs_auto.txt")

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--eprover_path", default="eprover")
    parser.add_argument("--eproverScheduleFile", default=os.path.expanduser("~/eprover/HEURISTICS/schedule.vars"))
    parser.add_argument("--soft_cpu_limit", type=int, default=1)
    parser.add_argument("--cpu_limit", type=int, default=2)

    parser.add_argument("--train_patience", type=int, default=1*2078, help="How many proof attempts to wait for another solved problem before stopping training.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problems = glob(f"{args.problems}/*")

    # This dashboard should only be updated by TrainPolicy or things it calls synchronously
    dashboard = rich_dashboard.DashBoard(f"Experiment Information for \"{args.run}\"", entropy_weight=args.entropy_weight, args=args)

    with open(args.cef_file) as f:
        CEF_STRING = f.read().replace("\n",'')

    if args.test:
        policy, opt = getPolicy(log=True)
        history = EvaluatePolicy(policy, problems, args)
        history.save(args.run, 1, eager=True)
        IPython.embed()
    else:
        history = TrainPolicy(problems, args)
        IPython.embed()

