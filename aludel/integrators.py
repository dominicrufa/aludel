"""custom integrators"""
import os
import sys
import openmm
from openmm import unit
import numpy as np
import copy
from typing import Any, Tuple, Dict, Iterable, Callable

class ThetaIntegratorV1(openmm.CustomIntegrator):
  """version 1 of a lambda dynamics integrator that performs lambda dynamics on `softcore_alpha`"""
  def __init__(self,
    softcore_alpha_str: str,
    include_theta_dynamics: bool=True,
    temperature: unit.Quantity=300*unit.kelvin,
    frictionCoeff: unit.Quantity = 1./unit.picoseconds,
    stepSize: unit.Quantity=2.*unit.femtoseconds,
    I: unit.Quantity = 5.*unit.amus*unit.nanometer**2,
    init_theta: float=np.pi/2.,
    **unused_kwargs):
    super().__init__(stepSize)
    self._theta_name = "theta_global"
    self._temperature = temperature
    self._frictionCoeff = frictionCoeff
    self._stepSize = stepSize
    self._I = I
    self._init_theta = init_theta
    self._kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    self._include_theta_dynamics = include_theta_dynamics

    self._add_global_variables(**unused_kwargs)
    self._add_body(**unused_kwargs)

  def _add_global_variables(self, **kwargs):
    self.addGlobalVariable("a", np.exp(-self._frictionCoeff*self._stepSize))
    self.addGlobalVariable("b", np.sqrt(1. - np.exp(-2.*self._frictionCoeff*self._stepSize)))
    self.addGlobalVariable("kT", self._kB*self._temperature)
    self.addGlobalVariable('x1', 0)
    self.addGlobalVariable("I",
      self._I.value_in_unit_system(unit.md_unit_system)) # moment of inertia of theta
    self.addGlobalVariable("omega", 0.) # randomize velocity
    self.randomize_omega(**unused_kwargs)
    self.addGlobalVariable(self._theta_name, self._init_theta) # initialize the value of theta

  def _add_full_V_step(self, **kwargs):
    self.addComputePerDof("v", "v + dt*f/m")
    self.addConstrainVelocities()
    if self._include_theta_dynamics:
      self.addComputeGlobal("omega",
        f"omega - dt*deriv(energy, {self._theta_name})/I")

  def _add_half_R_step(self, **kwargs):
    self.addComputePerDof("x", "x + 0.5*dt*v")
    if self._include_theta_dynamics:
      self.addComputeGlobal(self._theta_name, f"{self._theta_name} + 0.5*dt*omega")

  def _add_O_step(self, **kwargs):
    self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
    if self._include_theta_dynamics:
      self.addComputeGlobal("omega", "a*omega + b*sqrt(kT/I)*gaussian")

  def _add_constrain_R_fix_V(self, **kwargs):
    self.addComputePerDof("x1", "x")
    self.addConstrainPositions("v", "v + (x - x1)/dt")

  def _add_body(self, **kwargs):
    self.addUpdateContextState() # need to do this
    self._add_full_V_step(**kwargs)
    self._add_half_R_step(**kwargs)
    self._add_O_step(**kwargs)
    self._add_half_R_step(**kwargs)
    self._add_constrain_R_fix_V(**kwargs)

  def randomize_omega(self, **unused_kwargs):
    _val = np.sqrt(
      (self._kB*self._temperature/self._I).value_in_unit_system(unit.md_unit_system)
      )*np.random.normal()
    self.setGlobalParameterName('omega', _val)

class ThetaNonequilibriumIntegrator(ThetaIntegratorV1):
  """make a `ThetaIntegratorV1` with a Hamiltonian perturbation step on `lambda_global`"""
  def __init__(self, num_steps, lambda_global_name: str='lambda_global',
    *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._num_steps = num_steps
    self._lambda_global_name = lambda_global_name

  def _add_global_variables(self, **kwargs):
    super()._add_global_variables(**kwargs)
    self.addGlobalVariable('Eold', 0) #old energy value before perturbation
    self.addGlobalVariable('Enew', 0) #new energy value after perturbation
    self.addGlobalVariable('num_steps', self._num_steps)
    self.addGlobalVariable('current_step', 0.) # step counter for handling initialization and terminating integration
    self.addGlobalVariable(self._lambda_global_name, 'current_step / num_steps')
    self.addGlobalVariable('protocol_work', 0.)

  def _add_hamiltonian_pertubation_step(self, **unused_kwargs):
    self.addComputeGlobal('Eold', 'energy')
    self.addComputeGlobal('current_step', 'current_step + 1')
    self.addComputeGlobal(self._lambda_global_name, 'current_step / num_steps')
    self.addComputeGlobal('Enew', 'energy')
    self.addComputeGlobal('protocol_work', 'protocol_work + (Enew-Eold)')

  def _add_body(self, **kwargs):
    self.addUpdateContextState() # need to do this
    self._add_full_V_step(**kwargs)
    self._add_half_R_step(**kwargs)
    self._add_hamiltonian_pertubation_step(**kwargs)
    self._add_O_step(**kwargs)
    self._add_half_R_step(**kwargs)
    self._add_constrain_R_fix_V(**kwargs)
