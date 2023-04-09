# This script exists to automate the creation and destruction of tmux sessions.
# More specifically, this script enables the creation of panes running main.py
# for Cross Validation training

import sys, os
import argparse
import libtmux
from time import sleep
import itertools
from rich.progress import track
from rich import print


def killallSessions(server):
    try:
        for session in server.list_sessions():
            session.kill_session()
    except:
        print("Failed to kill all sessions...maybe none existed?")


def runCommandsInPane(commands, pane):
    for command in commands:
        pane.send_keys(command)


def getProblemsPath(args,i):
    # prefix = "~/Desktop/ATP/GCS/MPTPTP2078/Bushy/Folds"
    prefix = "~/Desktop/ATP/GCS/SLH-29/Folds"
    mode = "test" if args.test else "train"
    return os.path.expanduser(f"{prefix}/{i}/{mode}")


def nthMainCommands(args, i):
    trainRunName = args.run + f"{i}_train"
    testRunName = args.run + str(i)
    runName = testRunName if args.test else trainRunName

    special_args = [
        f"--problems \"{getProblemsPath(args,i)}\"",
        f"--run \"{runName}\"",
        f"--model_path \"models/{testRunName}.pt\"",
        f"--opt_path \"models/{testRunName}_opt.pt\"",
        "--test" if args.test else ""
    ]
    special_args = " ".join(special_args)
    
    commands = [
        "cd ~/Desktop/Reinforce\ E",
        "fish",
        f"python main.py {args.main_args} {special_args}",
    ]

    return commands



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--main_args")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--only_print", action="store_true")
    args = parser.parse_args()

    os.makedirs("models",exist_ok=True)

    if args.only_print:
        for i in range(5):
            print(f"Commands for pane {i}:")
            print(nthMainCommands(args,i))
            print("-"*90)
        sys.exit()

    server = libtmux.Server()
    session = server.new_session(args.run + ("" if args.test else "_train"))

    # Create Tmux Windows
    mode = "Testing" if args.test else "Training"
    session.list_windows()[0].rename_window(f"{mode} for CV fold 0 for {args.run}")
    for i in range(1,5):
        session.new_window(f"{mode} for CV fold {i} for {args.run}")
    
    # Run Commands in the windows
    windows = session.list_windows()
    for i,window in enumerate(track(windows)):
        pane = window.list_panes()[0]
        runCommandsInPane(nthMainCommands(args,i), pane)
        # break # Only for now...Comment this out later...
        sleep(0.5)

