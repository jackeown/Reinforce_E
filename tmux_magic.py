# This script exists to automate the creation and destruction of tmux sessions.
# More specifically, this script enables the creation of panes running e_caller.py and gain_xp.py
# with specific arguments.

import math
import argparse
import libtmux
from time import sleep
import itertools
from rich.progress import track


def killallSessions(server):
    try:
        for session in server.list_sessions():
            session.kill_session()
    except:
        print("Failed to kill all sessions...maybe none existed?")

def make4panes(window):
    window.split_window(attach=False, vertical=True)
    panes = window.list_panes()
    for pane in panes:
        pane.split_window(attach=False, vertical=False)
    
    panes = window.list_panes()
    return panes


def nthECallerCommands(n, run, eproverPath, host=None):
    commands = [
        "cd ~/Desktop/Reinforce\ E",
        "fish",
        f"python e_caller.py ~/Desktop/ATP/GCS/MPTPTP2078/Bushy/Problems/ --eproverPath {eproverPath} --num {n} --seed {n} --run {run} --total_workers {args.num_workers}",
    ]
    if host:
        commands = [f"ssh {host}"] + commands
    return commands

def nthGainXPCommands(n, run, host=None):
    commands = [
        "cd ~/Desktop/Reinforce\ E",
        "fish",
        f"python gain_xp.py --num {n}"
    ]

    if host:
        commands = [f"ssh {host}"] + commands
    return commands

def runCommandsInPane(commands, pane):
    for command in commands:
        pane.send_keys(command)


def getWorkerHosts(args):
    if args.worker_host_file:
        with open(args.worker_host_file) as f:
            return f.readlines()
    else:
        return [None]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("num_workers", type=int, default=4)
    parser.add_argument("--worker_host_file")
    parser.add_argument("--eproverPath", default="eprover")
    args = parser.parse_args()

    server = libtmux.Server()
    # killallSessions(server)

    session = server.new_session("workers")
    num_windows = math.ceil(args.num_workers / 2)
    for i in range(1, num_windows):
        session.new_window(str(i))
    windows = session.list_windows()

    hostGen = itertools.cycle(getWorkerHosts(args))

    current_worker = 0
    for window in track(windows):
        panes = make4panes(window)
        host = next(hostGen)
        for i in range(2):
            e_caller_commands = nthECallerCommands(current_worker + i, args.run, args.eproverPath, host)
            gain_xp_commands = nthGainXPCommands(current_worker + i, args.run, host)
            runCommandsInPane(e_caller_commands, panes[0 + 2*i])
            sleep(1.0)
            runCommandsInPane(gain_xp_commands, panes[1 + 2*i])
        current_worker += 2

