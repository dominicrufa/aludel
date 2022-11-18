
# coding: utf-8

# In[8]:


import os
import sys
import openmm
from openmm import unit
import numpy as np
import copy

from aludel.utils import decompress_pickle
from aludel.atm import SCRFSingleTopologyHybridSystemFactory
from aludel.atm import (get_hybrid_positions,
  get_original_positions_from_hybrid)
from openmmtools.constants import kB

# constants
temperature = 300*unit.kelvin
kT = (kB*temperature).value_in_unit_system(unit.md_unit_system)
default_theta_global = np.pi/2.
softcore_alpha_str = f"softcore_alpha = 0.25*cos(theta_global) + 0.5;"
theta_global_energy_str = f"-{kT}*log(sqrt(sin(theta_global)^2));"


# In[5]:


def translate_WaterBox_to_factory(direction:str = 'symmetric', **unused_kwargs):
  """direction is either
  1. symmetric: turn water 0 off and 1 on over lambda protocol
  2. off: turn water 0 off over protocol starting from on
  3. on: turn water 1 on over protocol starting from off"""
  from openmmtools.testsystems import WaterBox
  from copy import deepcopy
  wb = WaterBox()
  old_system = deepcopy(wb.system)
  new_system = deepcopy(wb.system)

  old_nbf = old_system.getForces()[2]
  new_nbf = old_system.getForces()[2]

  if direction == 'symmetric':
    _ = [old_nbf.setParticleParameters(i, 0., 1., 0.) for i in [3,4,5]] # turn water 1 off
    _ = [new_nbf.setParticleParameters(i, 0., 1., 0.) for i in [0,1,2]] # turn water 0 off
  elif direction == 'off':
    _ = [new_nbf.setParticleParameters(i, 0., 1., 0.) for i in [0,1,2]] # turn water 0 off
  elif direction == 'on':
    _ = [old_nbf.setParticleParameters(i, 0., 1., 0.) for i in [3,4,5]] # turn water 1 off
  else:
    raise Exception()
  
  unique_old_atoms = unique_new_atoms = []
  old_to_new_atom_map = {i:i for i in range(old_system.getNumParticles())}
  positions = wb.positions
  return old_system, new_system, unique_old_atoms, unique_new_atoms, old_to_new_atom_map, positions


# In[9]:


old_sys, new_sys, old_atoms, new_atoms, old_to_new, positions = translate_WaterBox_to_factory()


# In[10]:


from aludel.atm import ThetaIntegratorSCRFSingleTopologyHybridSystemFactory
stfactory = ThetaIntegratorSCRFSingleTopologyHybridSystemFactory(
default_theta_global=default_theta_global,
softcore_alpha_str = softcore_alpha_str,
theta_global_energy_str = theta_global_energy_str,
old_system=old_sys,
new_system=new_sys,
old_to_new_atom_map=old_to_new,
unique_old_atoms=old_atoms,
unique_new_atoms=new_atoms)

old_pos, new_pos = positions, positions

hybrid_system = stfactory.hybrid_system
hybrid_positions = get_hybrid_positions(old_positions=old_pos,
new_positions = new_pos, num_hybrid_particles=stfactory._hybrid_system.getNumParticles(),
old_to_hybrid_map = stfactory._old_to_hybrid_map,
new_to_hybrid_map = stfactory._new_to_hybrid_map)
old_pos = get_original_positions_from_hybrid(hybrid_positions, stfactory._hybrid_to_old_map)
new_pos = get_original_positions_from_hybrid(hybrid_positions, stfactory._hybrid_to_new_map)


# In[11]:


stfactory.test_energy_endstates(old_pos, new_pos, atol=1e-1, verbose=True)


# alright, now we can try to call the integrator.
