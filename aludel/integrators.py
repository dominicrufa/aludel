"""custom integrators"""
import os
import sys
import openmm
from openmm import unit
import numpy as np
import copy
from typing import from typing import Any, Tuple, Dict, Iterable, Callable

class ThetaIntegratorV1(openmm.CustomIntegrator):
  """version 1 of a lambda dynamics integrator that performs lambda dynamics on `lambda_global`"""
  def __init__(self, temperature: unit.Quantity=300*unit.kelvin,
    frictionCoeff: unit.Quantity = 1./unit.picoseconds,
    stepSize: unit.Quantity=2.*unit.femtoseconds,
    I: unit.Quantity = 5.*unit.amus*unit.nanometer**2,
    theta_name: str = "theta_global",
    init_theta_value: float = np.pi,
    init_lambda_f: Callable[float, float] = lambda _theta: 0.5*(np.cos(_theta)+1.),
    **unused_kwargs):
    super().__init__(stepSize)
    self.theta_name = theta_name
    self.I = I
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

    # Initialize
    self.addGlobalVariable("a", np.exp(-frictionCoeff*stepSize))
    self.addGlobalVariable("b", np.sqrt(1. - np.exp(-2.*frictionCoeff*stepSize)))
    self.addGlobalVariable("kT", kB*temperature)
    self.addGlobalVariable('x1', 0)
    self.addGlobalVariable("I",
      lambda_I.value_in_unit_system(unit.md_unit_system)) # moment of inertia of theta
    self.addGlobalVariable("omega", "sqrt(kT/I)*gaussian") # randomize velocity
    self.addGlobalVariable(theta_name, init_theta_value) # initialize the value of theta

    # Body

    self.addUpdateContextState() # need to do this
    # velocity updates
    self.addComputePerDof("v", "v + dt*f/m")
    self.addConstrainVelocities()
    self.addComputeGlobal("omega", f"omega - dt*deriv(energy, {theta_name})/I")
    # position updates
    self.addComputePerDof("x", "x + 0.5*dt*v")
    self.addComputeGlobal(theta_name, f"{theta_name} + 0.5*dt*omega")
    # randomize velocities
    self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
    self.addComputeGlobal("omega", "a*omega + b*sqrt(kT/I)*gaussian")
    # position update
    sef.addComputePerDof("x", "x + 0.5*dt*v")
    self.addComputeGlobal(theta_name, f"{theta_name} + 0.5*dt*omega")
    # constraints
    self.addComputePerDof("x1", "x")
    self.addConstrainPositions("v", "v + (x - x1)/dt")

  def randomize_omega(self, **unused_kwargs):
    """randomize the angular velocity of lambda_global
    (akin to `setVelocitiesToTemperature`)"""
    self.setGlobalVariableByName("omega", "sqrt(kT/I)*gaussian") # randomize velocity
