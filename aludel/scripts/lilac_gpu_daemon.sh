#!/bin/bash
#BSUB -P "rxn"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W  12:00
#BSUB -m "ly-gpu lx-gpu lu-gpu ld-gpu lt-gpu"
##BSUB -m "ls-gpu lg-gpu lt-gpu lp-gpu lg-gpu  ld-gpu"
#BSUB -o out_%I.stdout
##BSUB -cwd "/scratch/%U/%J"
#BSUB -eo out_%I.stderr
#BSUB -L /bin/bash

set -e
source ~/.bashrc
cd $LS_SUBCWD
#export PATH="/home/rufad/miniconda3/envs/openmm/bin:$PATH"
conda activate aludel
module load cuda/11.3
#python rxn_field_repex.py --data_path /data/chodera/rufad/aludel/tyk2/0_12.pbz2 --nc_prefix tyk2 --phase complex
