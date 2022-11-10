"""replica exchange simulations with `openmmtools`; mixins and stuff"""
import os
import openmm
from openmm import unit
from typing import Any, Tuple, Dict, Iterable, Callable, Tuple
import numpy as np
import copy

# openmmtools imports
from openmmtools.alchemy import AlchemicalState
from openmmtools import states
from openmmtools.states import CompoundThermodynamicState, SamplerState, ThermodynamicState
from openmmtools import cache
from openmmtools.multistate import replicaexchange

# specific to RF
from aludel.atm import SCRFSingleTopologyHybridSystemFactory

unit_type = type(unit)

# simulation-specific utilities
def minimize(thermodynamic_state : ThermodynamicState, sampler_state: SamplerState,
  tolerance: float, **unused_kwargs):
  # minimize and update sampler state in-place
  context, integrator = cache.global_context_cache.get_context(thermodynamic_state)
  sampler_state.apply_to_context(context, ignore_velocities = True)
  openmm.LocalEnergyMinimizer.minimize(context, tolerance=tolerance)
  sampler_state.update_from_context(context)

def get_openmm_platform(platform_name=None):
  """make a compliant platform for simulations"""
  if platform_name is None:
    from openmmtools.utils import get_fastest_platform
    platform = get_fastest_platform(minimum_precision='mixed')
  else:
    from openmm import Platform
    platform = Platform.getPlatformByName(platform_name)
  name = platform.getName()
  if name in ['CUDA', 'OpenCL']:
    platform.setPropertyDefaultValue('Precision', 'mixed')
  if name in ['CUDA']:
    platform.setPropertyDefaultValue('DeterministicForces', 'true')
    platform.setPropertyDefaultValue('CudaPrecision', 'mixed') # this just CUDA?
  return platform

# specific to `SCRFSingleTopologyHybridSystemFactory`
class SCRFSingleTopologyHybridLambdaProtocol(object):
  # lambda_protocol
  default_functions = {'lambda_global': lambda x: x}

  def __init__(self, *args, **kwargs):
    self.functions = copy.deepcopy(self.default_functions)

class RelativeAlchemicalState(AlchemicalState):
  class _LambdaParameter(AlchemicalState._LambdaParameter):
    pass
  lambda_global = _LambdaParameter('lambda_global')
  def set_alchemical_parameters(self, global_lambda, lambda_protocol):
    self.global_lambda=global_lambda
    for parameter_name in lambda_protocol.functions:
      lambda_val = lambda_protocol.functions[parameter_name](global_lambda)
      setattr(self, parameter_name, lambda_val)

class HybridCompatibilityMixin(object):
  """mixin for `MultiStateSampler`"""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def setup(self, system: openmm.System, n_states: int, positions: np.ndarray, temperature: openmm.unit,
    storage_file: str, minimization_tolerance: float=10, lambda_schedule: Iterable=None,
    lambda_protocol: Any=SCRFSingleTopologyHybridLambdaProtocol(), **unused_kwargs):

    # create alchemical state and protocol
    lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(system)
    thermostate = ThermodynamicState(system, temperature=temperature)
    compound_thermodynamic_state = CompoundThermodynamicState(thermostate,
    composable_states=[lambda_zero_alchemical_state])

    thermodynamic_state_list = []
    sampler_state_list = []

    if lambda_schedule is None:
      lambda_schedule = np.linspace(0, 1, n_states)

    sampler_state = SamplerState(positions,
      box_vectors=system.getDefaultPeriodicBoxVectors())

    for lambda_idx, lambda_val in enumerate(lambda_schedule):
      compound_thermodynamic_state_copy = copy.deepcopy(compound_thermodynamic_state)
      compound_thermodynamic_state_copy.set_alchemical_parameters(lambda_val, lambda_protocol)
      thermodynamic_state_list.append(compound_thermodynamic_state_copy)
      # minimize at appropriate state
      minimize(thermodynamic_state = compound_thermodynamic_state,
        sampler_state = sampler_state,
        tolerance = minimization_tolerance)

      # propagate at appropriate state (should I use a temperature ramp?)
      copy_move = copy.deepcopy(self._mcmc_moves)
      copy_move.apply(compound_thermodynamic_state_copy, sampler_state)
      sampler_state_list.append(copy.deepcopy(sampler_state))

    reporter = storage_file
    self.create(thermodynamic_states = thermodynamic_state_list,
      sampler_states=sampler_state_list, storage=reporter)

class HybridRepexSampler(HybridCompatibilityMixin, replicaexchange.ReplicaExchangeSampler):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

def get_f_from_nc(nc_path: str,
  **unused_kwargs) -> Tuple[unit_type, unit_type, np.array, np.array]:
  """compute the df and ddf matrices (in kT);
  returns (f, df, f_ij, df_ij) """
  from openmmtools.multistate import (MultiStateReporter,
    MultiStateSamplerAnalyzer)
  reporter = MultiStateReporter(nc_path)
  analyzer = MultiStateSamplerAnalyzer(reporter)
  f_ij, df_ij = analyzer.get_free_energy()
  f = f_ij[0,-1]*analyzer.kT
  df = df_ij[0,-1]*analyzer.kT
  out_f, out_df = (f.in_units_of(unit.kilocalorie_per_mole),
    df.in_units_of(unit.kilocalorie_per_mole))
  return out_f, out_df, f_ij, df_ij

def get_bindingdf_from_ncs(solvent_nc_path: str, complex_nc_path: str,
  **unused_kwargs) -> Tuple[unit_type, unit_type, Dict[str, Tuple[np.array, np.array]]]:
  """compute the binding df from solvent/complex legs;
  returns (binding_df, binding_ddf, {<phase>: f_ij, df_ij})"""
  metadata = {}
  data = []
  for _path, _phase in zip([solvent_nc_path, complex_nc_path],
    ['solvent', 'complex']):
    f, df, f_ij, df_ij = get_f_from_nc(_path)
    data.append([f, df])
    metadata[_phase] = (f_ij, df_ij)
  # solvent, then complex
  binding_df = data[0][0] - data[1][0]
  binding_ddf = np.sqrt(data[0][1]**2 + data[1][1]**2)
  return binding_df, binding_ddf, metadata
