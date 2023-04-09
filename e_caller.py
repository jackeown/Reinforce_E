from time import time, sleep
from glob import glob
from collections import defaultdict
import random
import re, os, sys, subprocess, multiprocessing, functools, itertools
from os.path import expanduser

from rich.progress import track
from rich import print
from rich.console import Console

import argparse
import IPython
import torch
from gain_xp import save_episode
from helpers import sendProbId, mean, Episode
import termplotlib as tpl
import scheduler_hack


def parseFolds(s):
    if s==None:
        return None

    try:
        folds = [int(x) for x in s.split(",")]
        return folds
    except:
        print("Failed to parse folds")
        exit()

def mkfifos(num):
    try:
        rw = 0o600
        os.mkfifo(f"/tmp/ProbIdPipe{num}", rw)
        os.mkfifo(f"/tmp/StatePipe{num}", rw)
        os.mkfifo(f"/tmp/ActionPipe{num}", rw)
        os.mkfifo(f"/tmp/RewardPipe{num}", rw)
        print("Made Named Pipes")
    except:
        print("Named Pipes Already Exist")

def extractStatus(stdout):
    try:
        return re.search("SZS status (.*)", stdout)[1]
    except:
        return "No SZS Status"

def extractProcessedCount(stdout):
    try:
        return int(re.search("# Processed clauses\s+: (.*)", stdout)[1])
    except:
        return "Error extracting processed count"

def extractOverhead(stdout):
    try:
        statePipeTime = re.search("# RL Seconds spent sending states\s*: (.*)", stdout)[1]
        actionPipeTime = re.search("# RL Seconds spent recieving actions\s*: (.*)", stdout)[1]
        rewardPipeTime = re.search("# RL Seconds spent sending rewards\s*: (.*)", stdout)[1]
        prepTime = re.search("# RL Seconds spent constructing 'state'\s*: (.*)", stdout)[1]
        return float(statePipeTime), float(actionPipeTime), float(rewardPipeTime), float(prepTime)
    except:
        return ["Error extracting overhead" for _ in range(4)]





def normalizeGiven(given):
    try:
        return re.search(r"cnf\([0-9a-z_]+, [a-z_]+, (.*)\)\.", given)[1]
    except:
        return None

def normalizePos(pos):
    try:
        return re.search(r"cnf\([0-9a-z_]+, [a-z_]+, (.*)\)\.", pos)[1]
    except:
        return None


def getGivenClauses(stdout):
    return [normalizeGiven(line) for line in stdout.split("\n") if "Given Clause" in line]

def getPositiveClauses(lines):
    return [normalizePos(line.split("# trainpos")[0]) for line in lines if "# trainpos" in line]

def getNegativeClauses(lines):
    return [normalizePos(line.split("#trainneg")[0]) for line in lines if "#trainneg" in line]



def getStates(stdout):
    try:
        matches = re.findall(r"RL State: (.*)\n", stdout)
        states = []
        for s in matches:
            try:
                states.append(eval(s))
            except:
                print(f"failed to eval '{s}' when parsing state")
        return states
    except:
        print("FAILED TO GET STATES!")
        return ['FAILURE_TO_REGEX']


def getActions(stdout):
    actions = ["FAILURE_TO_REGEX"]
    try:
        matches = re.findall("CEF Choice: ([0-9\.]*)\n", stdout)
        actions = [eval(s) for s in matches]
    except:
        print("FAILED TO GET ACTIONS")
    return actions



def varsFromLits(lits):
    return set(sum([re.findall(r"X[0-9]+",lit) for lit in lits], []))


def applyMapToLit(map, lit):
    placeholders = [f"__placeholder({i})__" for i in range(len(map))]
    for key,placeholder in zip(map.keys(), placeholders):
        lit = lit.replace(key,placeholder)

    for placeholder,val in zip(placeholders, map.values()):
        lit = lit.replace(placeholder, val)
    return lit

def allSubstitutions(literals, original_vars, new_vars):
    n = len(new_vars)

    for permutation in itertools.permutations(new_vars, n):
        map = dict(zip(original_vars, permutation))
        yield set([applyMapToLit(map,lit) for lit in literals])


def existsUnifyingSub(l1, l2):
    l1_vars = varsFromLits(l1)
    l2_vars = varsFromLits(l1)

    if len(l1_vars) != len(l2_vars):
        return False
    
    map1 = {k:"__placeholder__" for k in l1_vars}
    masked_l1 = set([applyMapToLit(map1, lit) for lit in l1])

    map2 = {k:"__placeholder__" for k in l2_vars}
    masked_l2 = set([applyMapToLit(map2, lit) for lit in l2])

    if masked_l1 != masked_l2:
        return False

    for subbed in allSubstitutions(l2, l2_vars, l1_vars):
        if l1 == subbed:
            return True
    return False


def litSetsEqual(lits1, lits2):

    if len(lits1) != len(lits2):
        return False

    if lits1 == lits2:
        return True

    return existsUnifyingSub(lits1, lits2)


def clause_in_set(g, lits_set):

    for lits in lits_set:
        if litSetsEqual(lits, g):
            return True

    return False




# def getRewards(stdout):
#     lines = stdout.split("\n")
    
#     given = getGivenClauses(stdout)

#     pos = getPositiveClauses(lines)
#     posClauses = [set(x[1:-1].split("|")) for x in pos]

#     rewards = [1 if g is not None and clause_in_set(set(g[1:-1].split("|")),posClauses) else 0 for g in given]

#     total = sum(rewards)
#     if total == 0:
#         return rewards

#     return [x / total for x in rewards]


def getRewards(stdout, n):

    rewards = set([int(x) for x in re.findall("# trainpos (-?[0-9]*)", stdout)])
    rewards = [1 if i in rewards else 0 for i in range(n)]

    total = sum(rewards)
    if total == 0:
        return rewards

    return [x / total for x in rewards]




# Override rich track so I can embed IPython
def track(*args, **kwargs):
    return args[0]


def getLatestModel(path):
    t = time()
    # If we've gotten it recently, just use that one
    if t - getLatestModel.lastUpdated < 1.0:
        return getLatestModel.model

    # Keep trying if getLatestModel.model is None
    tries = 0
    while getLatestModel.model is None or tries < 3:
        try:
            getLatestModel.model = torch.load(path)
            getLatestModel.lastUpdated = time()
            break
        except:
            sleep(1)

        tries += 1

    return getLatestModel.model

getLatestModel.lastUpdated = 0
getLatestModel.model = None

def getConfigName(stderr, yell=True):
    matches = re.findall(r"# Configuration: (.*)", stderr)
    if matches:
        return matches[-1]
    elif yell:
        print("NO CONFIG NAME!!!")

def parseHeuristicDef(heuristic):
    return {k:v for v,k in re.findall(r'([0-9]+)\.([^\)]+\))', heuristic)}

def loadConfigurations(pathToScheduleFile):
    with open(pathToScheduleFile) as f:
        lines = f.readlines()

    configs = {}
    for line in lines:
        if "heuristic_def" in line:
            key = re.findall(r'"([^\"]*)"', line)[0]
            value = re.findall(r'heuristic_def: \\"([^\\]+)', line)[0]
            configs[key] = parseHeuristicDef(value)

    return configs


def getConfig(stdout, stderr, configName=None, configurations=None, yell=True):
    if configName is None:
        configName = getConfigName(stderr+stdout, yell=yell)
    if configName is None:
        return None
    
    if configurations is None:
        if getConfig.configurations is not None:
            configurations = getConfig.configurations
        else:
            getConfig.configurations = loadConfigurations(args.eproverScheduleFile)
            configurations = getConfig.configurations
        
    return configurations[configName]
getConfig.configurations = None


def getCEFs(stdout, stderr, actions):
    try:
        configCEFs = list(getConfig(stdout, stderr).keys())
        return [configCEFs[i] for i in actions]
    except:
        IPython.embed()



# probIdPipe=None means that we're not using 
# any pipes at all and we're just calling E by itself.
def runE(prob, probIdPipe=None):
    info = {}

    env = os.environ.copy()
    env['eproverPath'] = args.eproverPath
    env["E_RL_STATEPIPE_PATH"] = f"/tmp/StatePipe{args.num}"
    env["E_RL_ACTIONPIPE_PATH"] = f"/tmp/ActionPipe{args.num}"
    env["E_RL_REWARDPIPE_PATH"] = f"/tmp/RewardPipe{args.num}"
    
    t1 = time()
    probId = int(prob[-8:-4]) # MPT0010+1.p for example becomes 10
    if probIdPipe is not None:
        sendProbId(probIdPipe, probId)

    p = subprocess.run(["./runE.sh", prob], shell=False, capture_output=True, env=env)
    t2 = time()
    stdout = p.stdout.decode("utf8")
    stderr = p.stderr.decode("utf8")
    info["status"] = extractStatus(stdout)

    if info['status'] == "No SZS Status":
        print("No szs status? sus")
        print("stdout: ",stdout[-3000:])
        print("stderr: ",stderr[-3000:])


    if "Segmentation fault" in stderr:
        print(f"SEGFAULT for {prob}")

    info["solved"] = ("Unsatisfiable" in info["status"]) or ("Theorem" in info["status"])
    info['states'] = getStates(stdout)
    info['actions'] = getActions(stdout)
    info['rewards'] = getRewards(stdout)
    info['configName'] = getConfigName(stderr)
    info['cefs'] = getConfig(stdout, stderr, info['configName'])

    
    info['args'] = args
    info["time"] = t2 - t1
    info["timestamp"] = t1
    info["processed_count"] = len(getGivenClauses(stdout))
    info["processed_count_reported"] = int(extractProcessedCount(stdout)) if info["solved"] else float('inf')
    info["problem"] = prob
    info["probId"] = probId
    info["statePipeTime"], info["actionPipeTime"], info["rewardPipeTime"], info["prepTime"] = extractOverhead(stdout)
    limit = 5_000
    info['stdout'] = "\n".join(stdout.split("\n")[-limit:])
    info['stderr'] = "\n".join(stderr.split("\n")[-limit:])

    a,b,c = len(info['states']), len(info['actions']), len(info['rewards'])
    if len({a,b,c}) != 1:
        print(f"Uh oh! the number of states, actions, and rewards are not equal: ({a},{b},{c})!")
        # IPython.embed()
    else:
        if len(info['states']) > 0:
            save_episode(Episode(
                prob,
                [torch.log(torch.tensor([x], dtype=torch.float) + 2.0) for x in info['states']],
                [torch.tensor([x]) for x in info['actions']],
                [torch.tensor([x]) for x in info['rewards']],
                getLatestModel(args.model_path) # THIS WILL BE THROWN AWAY BY NUMPY EPISODE SAVING!!!
            ), numpy=True)
    
    if not args.evaluate:
        print("Scheduler_hack removing problem")
        scheduler_hack.removeProblem(probId)

    return info





def whichProbsSolvable(problems, history, probIdPipe=None, parallel=False):
    random.shuffle(problems)
    solvable, unsolvable = [], []

    def processInfo(info, prob, save):
        if info['solved']:
                solvable.append(prob)
        else:
            unsolvable.append(prob)
        
        p = len(solvable) / (len(solvable) + len(unsolvable))
        print(f"{int(1000*p)/10}% solved")
        print(f"status: {info['status']}")
        history.addInfo(info)

        if save:
            history.save(args.run, args.num)

    if parallel:
        with multiprocessing.Pool(80) as p:
            infos = p.map(runE, problems)
        for info, prob in zip(infos, problems):
            processInfo(info, prob, save=False)
        history.save(args.run, args.num)
        
    else:
        for i, prob in enumerate(track(problems, description="Attempting to solve all problems")):
            print(f"Solving {prob}")
            info = runE(prob, probIdPipe)

            shouldSave = (not args.no_save_history) and ((i<20) or (i%10 == 0) or (i > len(problems)-3))
            processInfo(info, prob, save=shouldSave)
            

    print(f"Can solve {len(solvable)} problems")
    print(f"Average decrease in processed clauses: {history.avgProcCountDecrease()}")

    if args.evaluate:
        exit() # Just for evaluation...

    while not scheduler_hack.anyLeftToProve(allProblems):
        print("Finished. Waiting on other workers")
        sleep(10)
    
    return solvable, unsolvable



class ECallerHistory:

    def __init__(self):
        self.history = defaultdict(list)
        self.lastSaved = 0
    
    def probsSolved(self):
        return {x for x in self.history.keys() if self.getProofCount(x)}

    def merge(self, others):
        for other in others:
            h = other.history
            for key, value in other.history.items():
                self.history[key].extend(value)

    def addInfo(self, info):
        self.history[info['problem']].append(info)

    def getProofCount(self, prob):
        return len([1 for x in self.history[prob] if x['solved']])
    
    def getProofPercentage(self, prob):
        return self.getProofCount(prob) / len(self.history[prob])

    def getFirstProofInfo(self, prob):
        for info in self.history[prob]:
            if info['solved']:
                return info

    def getLatestProofInfo(self, prob):
        for info in reversed(self.history[prob]):
            if info['solved']:
                return info
    
    def getProcCounts(self, prob):
        return [info['processed_count'] for info in self.history[prob] if info['solved']]

    def getProcCountsReported(self, prob):
        return [info['processed_count_reported'] for info in self.history[prob] if info['solved']]

    def avgProcCountDecrease(self):
        decreases = []
        for prob in self.history:
            counts = self.getProcCounts(prob)
            if counts:
                decreases.append(counts[0] - counts[-1])
        return mean(decreases) if decreases else None

    def save(self, run, num, eager=True):

        if not eager and time() - self.lastSaved < 20*60:
            return

        os.makedirs(f"./ECallerHistory/{run}", exist_ok=True)
        torch.save(self, f"./ECallerHistory/{run}/{num}.history")

    @staticmethod
    def load(run, num=None):
        if num is None:
            histories = [torch.load(x) for x in glob(f"./ECallerHistory/{run}/*.history")]
        else:
            histories = [torch.load(f"./ECallerHistory/{run}/{num}.history")]
        x = ECallerHistory()
        x.merge(histories)
        return x

    @staticmethod
    def load_safe(run, num=None, progress=False):

        if not progress:
            f = lambda x: x
        else:
            f = track

        if num is None:
            histories = []
            for x in f(glob(f"./ECallerHistory/{run}/*.history")):
                while True:
                    try:
                        histories.append(torch.load(x))
                        break
                    except:
                        print(f"Failed to load {x}...retrying")
                        sleep(1)
        else:
            while True:
                try:
                    histories = [torch.load(f"./ECallerHistory/{run}/{num}.history")]
                    break
                except:
                    print("Failed to load history...retrying")
                    sleep(1)
        x = ECallerHistory()
        x.merge(histories)
        return x


    def learningGraph(self, smooth=2078):
        """Shows a termplotlib plot of proof percentage over time with the designated amount of smoothing"""
        all_infos = [info for infos in self.history.values() for info in infos]
        sorted_infos = sorted(all_infos, key=lambda x: x['timestamp'])
        sorted_solved = [1.0 if x['solved'] else 0.0 for x in sorted_infos]

        ys = [mean(sorted_solved[i:i+smooth]) for i in range(0, len(sorted_solved), smooth)]
        ys = ys[:-1]
        xs = list(range(len(ys)))

        fig = tpl.figure()
        fig.plot(xs,ys)
        return fig.get_string()



    def summarize(self, smooth=2078):
        """This method makes a really cool rich dashboard to show all relevant info in this history."""
        print(self.learningGraph(smooth))
        print(f"Average Processed Count Decrease across problems: {self.avgProcCountDecrease()}")
        # Show learningGraph
        # Show avgProcCountDecrease
        # Show ...



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problems_path")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--run", default="default", help="to help me remember what I've run when looking at saved history files.")
    parser.add_argument("--num", type=int, default=0, help="what is the index for this new worker? (Used for naming named pipes and other things)")
    parser.add_argument("--total_workers", type=int, default=1, help="for use with tmux_magic.py and the scheduler_hack")
    parser.add_argument("--which", choices=["all", "easy", "hard"], default="all", help="Should e_caller invoke E on easy problems more often or hard more often or simply always try to solve all problems?")
    parser.add_argument("--no_save_history", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--noPipe", action="store_true")
    parser.add_argument("--folds", default=None, type=parseFolds)
    parser.add_argument("--numFolds", default=5, type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", default="latest_model.pt")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--eproverPath", default='eprover')
    parser.add_argument("--eproverScheduleFile", default=expanduser("~/eprover/HEURISTICS/schedule.vars"))
    args = parser.parse_args()
    console = Console()


    allProblems = sorted(glob(f"{args.problems_path}/*.p"))
    random.seed(0)
    random.shuffle(allProblems)
    if args.folds is not None:
        n = len(allProblems)
        foldSize = n // args.numFolds
        folds = [allProblems[i:i+foldSize] for i in range(0, n, foldSize)]
        allProblems = set(sum([folds[i] for i in args.folds], []))
        args.run += f"_CVFolds_{''.join(str(x) for x in args.folds)}"

    allProblems = sorted(allProblems)
    random.seed(args.seed)

    if not args.evaluate and args.total_workers != 1:
        allProblems = [prob for prob in allProblems if int(prob[-8:-4]) % args.total_workers == args.num]

    if args.noPipe:
        probIdPipe = None
    else:
        mkfifos(args.num)
        probIdPipe = os.open(f"/tmp/ProbIdPipe{args.num}", os.O_WRONLY)


    history = ECallerHistory()
    if args.load:
        unsolvable = allProblems
        solvable = []
        history = ECallerHistory.load(args.run, args.num)
    else:
        solvable, unsolvable = whichProbsSolvable(allProblems, history, probIdPipe, args.parallel)
        if not args.no_save_history:
            history.save(args.run, args.num)




    ##############################
    # Only if not args.evaluate! #
    # |||||||||||||||||||||||||| #
    # vvvvvvvvvvvvvvvvvvvvvvvvvv #
    ##############################

    how_often_to_try_new_probs = 10
    epoch = 0
    while True:
        try:
            if args.which == "easy":
                for prob in track(solvable, description="Solving the solvable"):
                    info = runE(prob, probIdPipe)
                    history.addInfo(info)
                    
            elif args.which == "hard":
                for prob in track(unsolvable, description="Solving the unsolvable"):
                    info = runE(prob, probIdPipe)
                    history.addInfo(info)

            if epoch % how_often_to_try_new_probs == how_often_to_try_new_probs-1:
                solvable, unsolvable = whichProbsSolvable(allProblems, history, probIdPipe, args.parallel)
                if not args.no_save_history:
                    history.save(args.run, args.num)
            
            epoch += 1

        except KeyboardInterrupt:
            command = input("type 'exit' to exit: ")
            if command.lower() == "exit":
                sys.exit()
            IPython.embed()