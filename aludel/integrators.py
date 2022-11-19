"""custom integrators"""
import os
import sys
import openmm
from openmm import unit
import numpy as np
import copy
from typing import Any, Tuple, Dict, Iterable, Callable

class CustomLangevinMiddleIntegratorV2(openmm.CustomIntegrator):
  """
  make a custom langevin integrator that also integrates `theta`, which is equivalent to `theta_global`
  context global parameter.
  """
  def __init__(self,
    temperature: unit.Quantity, collision_rate: unit.Quantity, timestep: unit.Quantity,
    inertial_moment: unit.Quantity, initial_theta: float, **unused_kwargs):

    super().__init__(timestep)
    from openmmtools.constants import kB
    self._temperature = temperature
    self._collision_rate = collision_rate
    self._timestep = timestep
    self._inertial_moment = inertial_moment
    self._kB = kB
    self._initial_theta = initial_theta

    self._add_global_variables(**unused_kwargs)
    self._add_body(**unused_kwargs)

  def _add_global_variables(self, **unused_kwargs):
    self.addGlobalVariable("a", np.exp(-self._collision_rate*self._timestep));
    self.addGlobalVariable("b", np.sqrt(1-np.exp(-2*self._collision_rate*self._timestep)));
    self.addGlobalVariable("kT", self._kB*self._temperature);

    # for theta dynamics
    self.addGlobalVariable("I", self._inertial_moment.value_in_unit_system(unit.md_unit_system))
    self.addGlobalVariable("omega", 0.)
    self.addGlobalVariable("theta", self._initial_theta)
    self.addPerDofVariable("x1", 0);

  def _add_body(self, **unused_kwargs):
    self.addUpdateContextState()
    self._add_full_V(**unused_kwargs)
    self._add_half_R(**unused_kwargs)
    self._add_O(**unused_kwargs)
    self._add_half_R(**unused_kwargs)
    self._add_constrain_R_fix_V(**unused_kwargs)

  def _add_full_V(self, **unused_kwargs):
    self.addComputePerDof("v", "v + dt*f/m");
    self.addConstrainVelocities();
    self.addComputeGlobal("omega", "omega - dt*deriv(energy, theta_global)/I")

  def _add_half_R(self, **unused_kwargs):
    self.addComputePerDof("x", "x + 0.5*dt*v");
    self.addComputeGlobal("theta", "theta + 0.5*dt*omega")
    self.addComputeGlobal("theta_global", "theta")

  def _add_O(self, **unused_kwargs):
    self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian");
    self.addComputeGlobal("omega", "a*omega + b*sqrt(kT/I)*gaussian")

  def _add_constrain_R_fix_V(self, **unused_kwargs):
    self.addComputePerDof("x1", "x");
    self.addConstrainPositions();
    self.addComputePerDof("v", "v + (x-x1)/dt")

  def randomize_omega(self, **unused_kwargs):
    _val = np.sqrt(
      (self._kB*self._temperature/self._inertial_moment).value_in_unit_system(unit.md_unit_system)
      )*np.random.normal()
    self.setGlobalVariableByName('omega', _val)

class CustomNEQIntegrator(CustomLangevinMiddleIntegratorV2):
  """make a `ThetaIntegratorV1` with a Hamiltonian perturbation step on `lambda_global`"""
  def __init__(self, num_steps, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._num_steps = num_steps

  def _add_global_variables(self, **kwargs):
    super()._add_global_variables(**kwargs)
    self.addGlobalVariable('Eold', 0) #old energy value before perturbation
    self.addGlobalVariable('Enew', 0) #new energy value after perturbation
    self.addGlobalVariable('num_steps', self._num_steps)
    self.addGlobalVariable('current_step', 0.) # step counter for handling initialization and terminating integration
    self.addGlobalVariable('protocol_work', 0.)

  def _add_hamiltonian_pertubation_step(self, **unused_kwargs):
    self.addComputeGlobal('Eold', 'energy')
    self.addComputeGlobal('current_step', 'current_step + 1')
    self.addComputeGlobal('lambda_global', 'current_step / num_steps')
    self.addComputeGlobal('Enew', 'energy')
    self.addComputeGlobal('protocol_work', 'protocol_work + (Enew-Eold)')

  def _add_body(self, **unused_kwargs):
    self.addUpdateContextState()
    self._add_full_V(**unused_kwargs)
    self._add_half_R(**unused_kwargs)
    self._add_hamiltonian_pertubation_step(**unused_kwargs)
    self._add_O(**unused_kwargs)
    self._add_half_R(**unused_kwargs)
    self._add_constrain_R_fix_V(**unused_kwargs)
