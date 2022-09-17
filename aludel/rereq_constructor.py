"""given a `receptor.pdb` and `ligands.sdf` files, construct all of the
inputs of the hybrid system generator (hsg.py) and subsequent sampler objects
"""

# Imports
import openmm
from openmm import app, unit
import mdtraj as md
import numpy
import copy
from typing import Any, Tuple, Dict, Iterable, Callable

# DEFAULT_PHASE_PARAMETERS = {
# 'complex': ,
# 'solvent': ,
# }
# 
# DEFAULT_FORCEFIELD_FILES = [
#     'amber/ff14SB.xml',
#     'amber/tip3p_standard.xml',
#     'amber/tip3p_HFE_multivalent.xml',
#     ]
#
# # utilities
# def prepare_complex(receptor_pdbfile: str,
#                     guest_md_topology: str,
#                     spectator_pdb_filenames: Iterable[str]=None):
#     # receptor
#     receptor_pdbfile = open(receptor_pdbfile, 'r')
#     pdb_file = app.PDBFile(receptor_pdbfile)
#     receptor_pdbfile.close()
#     receptor_positions = pdb_file.positions
#     receptor_omm_topology = pdb_file.topology
#     receptor_md_topology = md.Topology.from_openmm(receptor_omm_topology)
#
#     complex_md_topology = receptor_md_topology.join(guest_md_topology)
#     n_atoms_spectators = 0
#     if spectator_pdb_filenames is not None:
#         for i, spectator_top in enumerate
#
#
#
# class SingleTopologyFEPSetup(object):
#     """
#     setup the single topology FEP generator.
#     """
#     def __init__(self,
#         ligand_sdf_filename: str,
#         receptor_pdb_filename: str,
#         old_new_ligand_indices: Iterable[int],
#         forcefield_files: Iterable[str]=DEFAULT_FORCEFIELD_FILES,
#         small_molecule_forcefield: str='openff-2.0.0',
#         spectator_filenames: str=None,
#         nonbonded_method: str='PME',
#         ):
