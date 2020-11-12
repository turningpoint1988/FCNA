#!/usr/bin/bash

Datadir=${1}
for experiment in $(ls ./$Datadir/)
do
    echo "working on $experiment."
    if [ ! -d ./models/$experiment ]; then
        mkdir ./models/$experiment
    else
        continue
    fi
   
    python run_motif.py -d `pwd`/$Datadir/$experiment/data \
                         -n $experiment \
                         -g 0 \
                         -b 100 \
                         -lr 0.001 \
                         -t 0.7 \
                         -e 20 \
                         -w 0.0005 \
                         -c `pwd`/models/$experiment
done
