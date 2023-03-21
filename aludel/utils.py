"""various utilities"""
import os
from openmm import unit
import openmm
from typing import List, Any, Set, Callable


def maybe_params_as_unitless(parameters: List) -> List:
    """translate an List of parameters (either unit.Quantity or not)
    into ints/floats"""
    dummy_unit = type(1. * unit.nanometers)
    outs = []
    for param in parameters:
        if type(param) == dummy_unit:
            outs.append(param.value_in_unit_system(unit.md_unit_system))
        else:
            outs.append(param)
    return outs


def handle_omissions(query_num_terms_method: Callable[[None], int], query_params_method: Callable[[int], List],
                     set_term_method: Callable[[List[Any], int], int], omission_sets: List[Set[int]],
                     parameter_replacement_list: List[Any] = None, **unused_kwargs):
    """query `openmm.Force` object and zero out any/all terms in place whose indices include a subset given by
    `omission_sets` return a dict of the term indices that were omitted and the returnable of `query_params_method`;
    the `parameter_replacement_list` is a list of term parameters that will replace the term params.

    Note: by default, the `parameter_replacement_list` is `None`, so we do not modify the force in-place, just query
    the term indices that _would_ be modified"""
    out_dict = {} # make an empty out dict
    is_replacement_none = parameter_replacement_list is None  # query whether the replacement is `None`
    if len(omission_sets) == 0: # if the omission set is empty, return by default
        return out_dict
    if len(omission_sets) == 1: # if the omission set only has 1 entry, get the set
        interest_indices = omission_sets[0]
    else: # make a set of all the entries
        interest_indices = omission_sets[0].union(*omission_sets[1:])

    num_terms = query_num_terms_method()
    for term_idx in range(num_terms): # iterate over the term indices
        all_params = query_params_method(term_idx) # query all the params
        particle_indices = all_params[:-1] # get the hybrid particle indices
        if len(interest_indices.union(set(particle_indices))) > 0: # if there is a match somewhere, go to more logic
            params = all_params[-1] if is_replacement_none else parameter_replacement_list # generate replacement params
            truths = [_set.issubset(set(particle_indices)) for _set in omission_sets] # ask if we are a subset
            if any(truths): # if there is a match, record and mod the params
                out_dict[term_idx] = all_params
                _ = set_term_method(*particle_indices, params)
    return out_dict



def sort_indices_to_str(indices: List[int]) -> str:
    sorted_indices = sorted(indices)
    return '.'.join([str(_q) for _q in sorted_indices])


def compressed_pickle(filename: str, data: Any):
    # pickle a file and compress
    import bz2
    import _pickle as cPickle
    with bz2.BZ2File(filename, 'w') as f:
        cPickle.dump(data, f)


def decompress_pickle(filename: str) -> Any:
    # Load any compressed pickle file
    import bz2
    import _pickle as cPickle
    data = bz2.BZ2File(filename, 'rb')
    data = cPickle.load(data)
    return data


def read_pickle(filename: str) -> Any:
    import pickle
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def deserialize_xml(xml_filename):
    """
    load and deserialize an xml
    arguments
        xml_filename : str
            full path of the xml filename
    returns
        xml_deserialized : deserialized xml object
    """
    from openmm.openmm import XmlSerializer
    with open(xml_filename, 'r') as infile:
        xml_readable = infile.read()
    xml_deserialized = XmlSerializer.deserialize(xml_readable)
    return xml_deserialized


def serialize_xml(object, xml_filename):
    """
    load and deserialize an xml
    arguments
        object : object
            serializable
        xml_filename : str
            full path of the xml filename
    """
    from openmm.openmm import XmlSerializer
    with open(xml_filename, 'w') as outfile:
        serial = XmlSerializer.serialize(object)
        outfile.write(serial)


def query_outdirs_from_perses(
        perses_base_dir: str,
        write_to_dir: str,
        outdir_prefix: str = 'out_',
        specific_query_dirs: List[str] = [],
        out_topology_proposal_name: str = 'out-topology_proposals.pkl'):
    """
    a simple utility to extract the inputs to the `hsg.py` from a canonical
    `perses` execution array.
    """
    pref_len = len(outdir_prefix)
    if not os.path.isdir(write_to_dir):  # make the dir if doesn't exists
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
                out_data[phase][f"{_key}_positions"] = _data[
                    f"{phase}_{_key}_positions"]

                # unique atoms
                out_data[phase][f"unique_{_key}_atoms"] = getattr(top_proposal,
                                                                  f"_unique_{_key}_atoms")

                # system
                out_data[phase][f"{_key}_system"] = getattr(top_proposal,
                                                            f"_{_key}_system")
            out_data[phase][f"old_to_new_atom_map"] = getattr(top_proposal,
                                                              'old_to_new_atom_map')
        return out_data

    if len(specific_query_dirs) > 0:
        query_dir_List = specific_query_dirs
    else:
        query_dir_List = []
        query_dir_List_try = [_q for _q in
                              os.listdir(perses_base_dir) if os.path.isdir(
                os.path.join(perses_base_dir, _q))]
        for _dir_List in query_dir_List_try:
            if len(_dir_List) > pref_len:
                if _dir_List[:pref_len] == outdir_prefix:
                    query_dir_List.append(_dir_List)

    for _dirpath in query_dir_List:
        _query_pathname = os.path.join(perses_base_dir, _dirpath)
        _out_dict = _query_singular_dir(_query_pathname)
        compressed_pickle(os.path.join(
            write_to_dir, _dirpath[pref_len:] + ".pbz2"), _out_dict)
