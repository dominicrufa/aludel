"""
run equilibrium simulation of rxn field/alchemical water
(i.e. `aludel.temp.softcore_water`) at 3 flavours.
"""

import os
import sys
import openmm
from openmm import unit
import numpy as np
import copy
import argparse
from aludel.temp import softcore_water
from aludel.repex import get_openmm_platform
from aludel.utils import serialize_xml


parser = argparse.ArgumentParser(description = f"run water equil.")
parser.add_argument('--direction', type=str, default=None) # 'symmetric', 'off', 'on'
args = parser.parse_args()

if __name__ == "__main__":
  platform = get_openmm_platform('CUDA')
  factory, positions = softcore_water.translate_WaterBox_to_factory(
    direction = args.direction)
  # minimize
  integrator = openmm.LangevinMiddleIntegrator(softcore_water.temperature,
    1./unit.picoseconds, 4.*unit.femtoseconds)
  context = openmm.Context(factory.hybrid_system, integrator, platform)
  out_cache = {0.: {'box_vectors': [], 'positions': [], 'pe': []},
               1.: {'box_vectors': [], 'positions': [], 'pe': []}}
  for key, in out_cache.keys():
    context.setParameter('lambda_global', key)
    openmm.LocalEnergyMinimizer.minimize(context, tolerance=10.)
    context.setVelocitiesToTemperature(softcore_water.temperature)
    for i in range(200):
      integrator.step(2500) # that's 10ps
      state = context.getState(getEnergy=True, getPositions=True)
      out_cache[key]['box_vectors'].append(
        state.getPeriodicBoxVectors(asNumpy=True).value_in_unit_system(unit.md_unit_system))
      out_cache[key]['positions'].append(
        state.getPositions(asNumpy=True).value_in_unit_system(unit.md_unit_system))
      out_cache[key]['pe'].append(
        state.getPotentialEnergy().value_in_unit_system(unit.md_unit_system))
  # now save
  for key in out_cache.keys():
    for _key in out_cache[key].keys():
      out_cache[key][_key] = np.array(out_cache[key][_key])
  np.savez(f"{args.direction}.water.data.npz", out_cache)
  serialize_xml(factory.hybrid_system, f"{args.direction}.water.system.xml")
