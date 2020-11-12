#!/usr/bin/bash

Data=${1}
model=${2}
for experiment in $(ls ./${Data}/)
do
    echo "working on ${experiment}."
    if [ ! -d ./motifs/${experiment} ]; then
        mkdir ./motifs/${experiment}
    else
        continue
    fi
    
    python motif_finder.py -d `pwd`/${Data}/${experiment}/data \
                           -n ${experiment} \
                           -t 0.7 \
                           -g 0 \
                           -c `pwd`/$model/${experiment} \
                           -o `pwd`/motifs/${experiment}
done
