"""script and submission utilities"""
import numpy as np
import typing
import copy

DEFAULT_BSUB_STR = \
"""
#!/bin/bash
#BSUB -P {job_name}
#BSUB -n {num_cores}
#BSUB -R rusage[mem={memory}]
#BSUB -R span[hosts={hosts}]
#BSUB -q {queue}
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W {time_to_run}
#BSUB -m {nodes}
#BSUB -o {job_name}.stdout
#BSUB -eo {job_name}.stderr
#BSUB -L /bin/bash
set -e 
source ~/.bashrc
cd $LS_SUBCWD
{conda_version} activate {conda_env}
"""

def make_lilac_submit_script(
    command: str,
    name_prefix: str = 'run',
    job_name: str = 'run',
    num_cores: int = 1,
    memory: int = 8,
    hosts: int=1,
    queue: str = 'gpuqueue',
    time_to_run: str = '8:00',
    nodes: str = "'lj-gpu ll-gpu ln-gpu ly-gpu lx-gpu lu-gpu ld-gpu'",
    conda_version: str = 'micromamba',
    conda_env: str = 'dc',
    **unused_kwargs
    ) -> str:
    """make a lilac submission script which can be submitted with `bsub < {name_prefix}.sh`;
    EXAMPLE:
    >>> make_lilac_submit_script(
    >>>    "print('hello world'); import openmm; platform = openmm.Platform.getPlatformByName('CUDA'); print(platform.getName())"
    >>>     )"""
    bsub_str = DEFAULT_BSUB_STR.format(**locals())
    full_code = f"""{bsub_str} \n{command} """
    with open(f"{name_prefix}.sh", 'w') as f:
        f.write(full_code)
