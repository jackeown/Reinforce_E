from collections import defaultdict
from email.policy import default
from re import S
from time import time
from copy import deepcopy
from rich import print

class Profiler:
    def __init__(self):
        self.times = defaultdict(list)

    def copy(self):
        p = Profiler()
        for k,v in self.times.items():
            p.times[k] = deepcopy(v)
        return p
    
    def reset(self):
        self.times = defaultdict(list)

    def profile(self, tag):
        return ProfilerContextManager(self, tag)
    
    def report(self, as_str=False, rich=True):
        items = self.times.items()
        items = sorted(items, key=lambda x:sum(x[1]))

        s = ""
        for key,val in items:
            key = f"{key:40}: "
            times = f"mean={sum(val)/len(val):7.4f} sum={sum(val):7.4f}\n"
            if rich:
                key = f"[cyan]{key}[/cyan]"
                times = f"[white]{times}[/white]"

            s += f"{key}{times}"
        
        s = "#"*79 + "\n" + s + "#"*79

        if as_str:
            return s
        print(s)
    
    def merge(self, otherProfiler):
        """This method merges information from 'otherProfiler' into this one.
        The list of times for every key from both profilers will be present, with 
        lists for the same key being merged."""
        for k,v in otherProfiler.times.items():
            if k in self.times:
                self.times[k].extend(v)
            else:
                self.times[k] = v


class ProfilerContextManager:
    def __init__(self, profiler, tag):
        self.profiler = profiler
        self.tag = tag

    def __enter__(self):
        self.t1 = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.times[self.tag].append(time() - self.t1)



# Usage:

# my_profiler = Profiler()
# with my_profiler.profile("Identifier"):
#     # Do some stuff
#     pass


# my_profiler.report() # will show average and sum of time spent in "Identifier" across multiple invocations...
