
import random
import sys
from time import sleep

from glob import glob
import torch



while True:

    try:
        sleep(1.0)
        
        eps = glob("episodes/*.episode")
        ep = torch.load(random.choice(eps))

        for path in sys.argv[1:]:
            policy_net = torch.load(path)

            state = random.choice(ep.states)
            out = policy_net(state)

            soft = torch.softmax(out,1)

            print(soft)
            # print((soft*10000).round() / 100)
            print(torch.argsort(soft)[0][-10:])
    except:
        sleep(1.0)
