"""various utilities"""
import pickle
import os
from typing import Iterable


def compressed_pickle(filename:str, data:Any):
  # pickle a file and compress
  import bz2
  import _pickle as cPickle
  with bz2.BZ2File(filename, 'w') as f:
    cPickle.dump(data, f)

def decompress_pickle(filename:str) -> Any:
  # Load any compressed pickle file
  data = bz2.BZ2File(filename, 'rb')
  data = cPickle.load(data)
  return data

def read_pickle(filename:str) -> Any:
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def query_outdirs_from_perses(
    perses_base_dir: str,
    write_to_dir: str,
    outdir_prefix: str='out_',
    specific_query_dirs = Iterable[str]=[],
    out_topology_proposal_name: str='out-topology_proposals.pkl'):
    """
    a simple utility to extract the inputs to the `hsg.py` from a canonical
    `perses` execution array.
    """
    pref_len = len(outdir_prefix)
    if not os.path.isdir(write_to_dir): # make the dir if doesn't exists
        os.makedirs(write_to_dir)

    def _query_singular_dir(dir_path):
        """from a dir path, extract the old/new positions, unique atoms,
        systems, and `old_to_hybrid_map`"""
        out_data = {'complex': {}, 'solvent': {}}
        file_to_query = os.path.join(dir_path, out_topology_proposal_name)
        _data = read_pickle(file_to_query)
        for phase in out_data.keys():
            top_proposal = _data[f"{phase}_topology_proposal"]
            for _key in ['old', 'new']:
                # positions
                out_data[phase][f"{_key}_positions"] = /
                    _data[f"{phase}_{_key}_positions"]

                # unique atoms
                out_data[phase][f"unique_{_key}_atoms"] = /
                    getattr(top_proposal, f"_unique_{_key}_atoms")

                # system
                out_data[phase][f"{_key}_system"] = /
                    getattr(top_proposal, f"_{_key}_system")
            out_data[phase][f"old_to_new_atom_map"] = /
                top_proposal.old_to_new_atom_map
        return out_data

    if len(specific_query_dirs) > 0:
        query_dir_iterable = specific_query_dirs
    else:
        query_dir_iterable = []
        query_dir_iterable_try = [_q for _q in /
            os.listdir(perses_base_dir) if os.path.isdir(
            os.path.join(perses_base_dir, _q))]
        for _dir_iterable in query_dir_iterable_try:
            if len(_dir_iterable) > pref_len:
                if _dir_iterable[:pref_len] == outdir_prefix:
                    query_dir_iterable.append(_dir_iterable)

    for _dirpath in query_dir_iterable:
        _query_pathname = os.path.join(perses_base_dir, _dirpath)
        _out_dict = _query_singular_dir(_query_pathname)
        compressed_pickle(os.path.join(
        write_to_dir, _dirpath[pref_len:] + ".pbz2"), _out_dict)
