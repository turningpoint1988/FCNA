#!/usr/bin/bash

Datadir=${1}
model=${2}
for experiment in $(ls ./$Datadir/)
do
    echo "working on $experiment."
    if [ ! -d ./Refine/$experiment ]; then
        mkdir ./Refine/$experiment
    fi
   
    python locate_motif.py -d `pwd`/$Datadir/$experiment/data \
                         -n $experiment \
                         -g 0 \
                         -t 0.9 \
                         -c `pwd`/$model/$experiment \
                         -o `pwd`/Refine/$experiment
done
