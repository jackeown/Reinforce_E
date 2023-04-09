#!/bin/bash

export E_RL_STATEPIPE_PATH=$E_RL_STATEPIPE_PATH
export E_RL_ACTIONPIPE_PATH=$E_RL_ACTIONPIPE_PATH
export E_RL_REWARDPIPE_PATH=$E_RL_REWARDPIPE_PATH

SOFT=40
HARD=45

NORMAL_FLAGS="-l1 --print-statistics --training-examples=3 --soft-cpu-limit=$SOFT --cpu-limit=$HARD"
MAGIC_FLAGS="--simul-paramod --forward-context-sr --strong-destructive-er --destructive-er-aggressive --destructive-er --presat-simplify -F1 -WSelectComplexExceptUniqMaxHorn -tKBO6 -winvfreqrank -c1 -Ginvfreq --strong-rw-inst"

#x=`cat cefs.txt`
x=`cat cefs_auto.txt`
#x=`cat cefs_constcat_distilled.txt`
#x=`cat cefs_auto_weighted.txt`
# x=`cat cefs_small.txt`
# x=`cat cefs_7.txt`
# x=`cat cefs_smart_chosen.txt`

$eproverPath $NORMAL_FLAGS $MAGIC_FLAGS -H"$x" $1
#$eproverPath --auto $NORMAL_FLAGS $1
