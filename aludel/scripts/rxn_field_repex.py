"""execute repex reaction field script"""
# generic imports
import os
import sys
import openmm
from openmm import unit
import numpy as np
import copy
import argparse

# aludel imports
from aludel.repex import HybridRepexSampler
from aludel.utils import decompress_pickle, get_openmm_platform
from aludel.atm import SCRFSingleTopologyHybridSystemFactory
from aludel.atm import (get_hybrid_positions,
  get_original_positions_from_hybrid)
from openmmtools import cache, utils, mcmc

parser = argparse.ArgumentParser(description = f"run rf repex...")
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--nc_prefix', type=str, default=None)
parser.add_argument('--phase', type=str, default=None)
parser.add_argument('--num_iters', type=int, default=5000)
args = parser.parse_args()

if __name__ == "__main__":
  data = decompress_pickle(args.data_path)
  pbz2_filename = '/'.split(args.data_path)[-1]
  pbz2_filename_prefix = '.'.split(pbz2_filename)[0]
  out_ncfile_base = args.nc_prefix + '.' + pbz2_filename_prefix + '.nc'

  stfactory = SCRFSingleTopologyHybridSystemFactory(
  old_system=data[args.phase]['old_system'],
    new_system = data[args.phase]['new_system'],
    old_to_new_atom_map=data[args.phase]['old_to_new_atom_map'],
    unique_old_atoms=data[args.phase]['unique_old_atoms'],
    unique_new_atoms=data[args.phase]['unique_new_atoms'])

  old_pos, new_pos = data[args.phase]['old_positions'], data[args.phase]['new_positions']

  hybrid_system = stfactory.hybrid_system
  hybrid_positions = get_hybrid_positions(old_positions=old_pos,
  new_positions = new_pos, num_hybrid_particles=stfactory._hybrid_system.getNumParticles(),
    old_to_hybrid_map = stfactory._old_to_hybrid_map,
    new_to_hybrid_map = stfactory._new_to_hybrid_map)
  hybrid_old_positions = get_original_positions_from_hybrid(hybrid_positions,
    stfactory._hybrid_to_old_map)

  platform = get_openmm_platform() # make the platform appropriately

  mcmc_move = mcmc.LangevinDynamicsMove(timestep=2.*unit.femtoseconds,
    collision_rate = 1./unit.picoseconds, n_steps=500, reassign_velocities=True,
    n_restart_attempts=20, constraint_tolerance=1e-6)

  sampler = HybridRepexSampler(mcmc_moves=mcmc_move, online_analysis_interval=10)

  reporter = MultiStateReporter(out_ncfile_base,
    analysis_particle_indices=list(range(100)), checkpoint_interval=100)
  sampler.setup(system = hybrid_system, n_states=11, positions=hybrid_positions,
    temperature=300*unit.kelvin, storage_file=reporter,
    minimization_tolerance=10, lambda_schedule=np.linspace(0, 1., 11))

  # set the context cache platforms appropriately
  hss.energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None,
    platform=platform)
  hss.sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None,
    platform=platform)

  sampler.equilibrate(10) # this is a bit arbitrary
  sampler.extend(args.num_iters) # this is not arbitrary
