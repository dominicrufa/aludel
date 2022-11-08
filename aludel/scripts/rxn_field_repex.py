"""execute repex reaction field script"""
# generic imports
import os
import sys
import openmm
from openmm import unit
import numpy as np
import copy

# aludel imports
from aludel.repex import HybridRepexSampler
from aludel.utils import decompress_pickle
from aludel.atm import SCRFSingleTopologyHybridSystemFactory
from aludel.atm import (get_hybrid_positions,
  get_original_positions_from_hybrid)

phase = 'solvent'

data = decompress_pickle('../github/aludel/aludel/data/0_12.pbz2')

stfactory = SCRFSingleTopologyHybridSystemFactory(
old_system=data[phase]['old_system'],
  new_system = data[phase]['new_system'],
  old_to_new_atom_map=data[phase]['old_to_new_atom_map'],
  unique_old_atoms=data[phase]['unique_old_atoms'],
  unique_new_atoms=data[phase]['unique_new_atoms'],
  cutoff=1.2)

old_pos, new_pos = data[phase]['old_positions'], data[phase]['new_positions']

hybrid_system = stfactory.hybrid_system
hybrid_positions = get_hybrid_positions(old_positions=old_pos,
new_positions = new_pos, num_hybrid_particles=stfactory._hybrid_system.getNumParticles(),
  old_to_hybrid_map = stfactory._old_to_hybrid_map, new_to_hybrid_map = stfactory._new_to_hybrid_map)
hybrid_old_positions = get_original_positions_from_hybrid(hybrid_positions, stfactory._hybrid_to_old_map)

platform = get_openmm_platform()

from openmmtools import mcmc
mcmc_move = mcmc.LangevinDynamicsMove(timestep=2.*unit.femtoseconds,
  collision_rate = 1./unit.picoseconds, n_steps=500, reassign_velocities=True,
  n_restart_attempts=20, constraint_tolerance=1e-6)

sampler = HybridRepexSampler(mcmc_moves=mcmc_move, online_analysis_interval=10)
reporter = MultiStateReporter("test.nc", analysis_particle_indices=list(range(100)),
  checkpoint_interval=100)
sampler.setup(system = hybrid_system, n_states=11, positions=hybrid_positions,
  temperature=300*unit.kelvin, storage_file=reporter,
  minimization_tolerance=10, lambda_schedule=np.linspace(0, 1., 11))

sampler.equilibrate(10)
sampler.extend(10)
