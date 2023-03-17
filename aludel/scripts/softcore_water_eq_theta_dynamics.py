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
from aludel.integrators import ThetaIntegratorV1

platform = get_openmm_platform('CUDA')
factory, positions = softcore_water.translate_WaterBox_to_factory(
  direction = args.direction)
integrator = openmm.ThetaIntegratorV1(
  softcore_alpha_str = softcore_water.softcore_alpha_str,
  include_theta_dynamics = False,
  temperature = softcore_water.temperature,
  frictionCoeff = 1./unit.picoseconds,
  stepSize = 4.*unit.femtoseconds)
  context = openmm.Context(factory.hybrid_system, integrator, platform)
  context.setPositions(positions)
  context.setParameter('lambda_global', 0.)
  openmm.LocalEnergyMinimizer.minimize(context, tolerance=10.)
  context.setVelocitiesToTemperature(softcore_water.temperature)
  # now save
  for key in out_cache.keys():
    for _key in out_cache[key].keys():
      out_cache[key][_key] = np.array(out_cache[key][_key])
  np.savez(f"{args.direction}.water.data.npz", out_cache)
  serialize_xml(factory.hybrid_system, f"{args.direction}.water.system.xml")
