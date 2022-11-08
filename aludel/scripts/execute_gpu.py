import os
import sys

data_path = '/data/chodera/rufad/aludel/tyk2'
nc_prefix = 'tyk2'
run_template = "python rxn_field_repex.py --data_path {data_path} --nc_prefix {nc_prefix} --phase {phase}"
all_files = os.listdir(data_path)
all_pbz2s = [i for i in all_files if i[-4:] == 'pbz2']
all_pbz2_paths = [os.path.join(data_path, i) for i in all_pbz2s]
print(f"all files: {all_pbz2_paths}")
for i in all_pbz2_paths:
  in_data_path = os.path.join(data_path, i)
  for phase in ['solvent', 'complex']:
    _links = run_template.format(data_path=in_data_path, nc_prefix=nc_prefix, phase=phase)
    os.system("cp lilac_gpu_daemon.sh _.sh")
    os.system(f"echo '{_links}' >> _.sh")
    os.system("bsub < _.sh")
    os.system("rm _.sh")
