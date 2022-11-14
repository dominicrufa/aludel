"""reaction field conversion"""

import openmm
from openmm import app, unit
import numpy as np
import copy
from typing import Any, Tuple, Dict, Iterable, Callable
from aludel.utils import maybe_params_as_unitless

class ReactionFieldConverter(object):
    """
    convert a canonical `openmm.System` object's `openmm.NonbondedForce`
    to a `openmm.CustomNonbondedForce` that treats electrostatics with reaction
    field;
    see: 10.1039/d0cp03835k;
    adapted from
    https://github.com/rinikerlab/reeds/blob/\
    52882d7e009b5393df172dd4b703323f1d84dabb/reeds/openmm/reeds_openmm.py#L265
    """
    def __init__(self,
                 system : openmm.System,
                 cutoff: float=1.2,
                 eps_rf: float=78.5,
                 ONE_4PI_EPS0: float=138.93545764438198,
                 **unused_kwargs,
                 ):
        """
        It is assumed that the nonbonded force is "canonical"
        in that it contains N particles and N_e exceptions
        without further decorating attrs.

        Args:
            system : openmm.System
            cutoff : float=1.2 (cutoff in nm)
            eps_rf : float=78.5; dielectric constant of solvent
        """
        import copy

        nbfs = [f for f in system.getForces() if f.__class__.__name__ \
            == "NonbondedForce"]
        assert len(nbfs) == 1, f"{len(nbfs)} nonbonded forces were found"

        self._nbf = nbfs[0]
        self._system = system
        self._cutoff = cutoff
        self._eps_rf = eps_rf
        self.ONE_4PI_EPS0 = ONE_4PI_EPS0

        pair_nbf = self.handle_nonbonded_pairs()
        exception_bf = self.handle_nb_exceptions()
        self_bf = self.handle_self_term()

    @property
    def rf_system(self):
        import copy
        new_system = copy.deepcopy(self._system)

        pair_nbf = self.handle_nonbonded_pairs()
        exception_bf = self.handle_nb_exceptions()
        self_bf = self.handle_self_term()

        # remove the nbf altogether
        for idx, force in enumerate(self._system.getForces()):
            if force.__class__.__name__ == 'NonbondedForce':
                break
        new_system.removeForce(idx)

        for force in [pair_nbf, exception_bf, self_bf]:
            new_system.addForce(force)
        return new_system

    def handle_nonbonded_pairs(self):
        energy_fn = self._get_energy_fn()
        energy_fn += f"chargeprod_ = charge1 * charge2;"

        custom_nb_force = openmm.CustomNonbondedForce(energy_fn)
        custom_nb_force.addPerParticleParameter('charge') # always add

        custom_nb_force.addPerParticleParameter('sigma')
        custom_nb_force.addPerParticleParameter('epsilon')
        custom_nb_force.setNonbondedMethod(openmm.CustomNonbondedForce.\
            CutoffPeriodic) # always
        custom_nb_force.setCutoffDistance(self._cutoff)
        custom_nb_force.setUseLongRangeCorrection(False) # for lj, never

        # add particles
        for idx in range(self._nbf.getNumParticles()):
            c, s, e = self._nbf.getParticleParameters(idx)
            custom_nb_force.addParticle([c, s, e])

        # add exclusions from nbf exceptions
        for idx in range(self._nbf.getNumExceptions()):
            j, k, _, _, _ = self._nbf.getExceptionParameters(idx)
            custom_nb_force.addExclusion(j,k)
        return custom_nb_force

    def handle_nb_exceptions(self):
        energy_fn = self._get_energy_fn(exception=True)
        custom_b_force = openmm.CustomBondForce(energy_fn)
        # add terms separately so we need not reimplement the energy fn
        for _param in ['chargeprod', 'sigma', 'epsilon', 'chargeprod_']:
            custom_b_force.addPerBondParameter(_param)

        # copy exceptions
        for idx in range(self._nbf.getNumExceptions()):
            j, k , chargeprod, mix_sigma, mix_epsilon = self._nbf.\
                getExceptionParameters(idx)

            # now query charges, sigma, epsilon
            c1, _, _ = self._nbf.getParticleParameters(j)
            c2, _, _ = self._nbf.getParticleParameters(k)

            custom_b_force.addBond(j, k,
                                [chargeprod, mix_sigma, mix_epsilon, c1*c2])

        return custom_b_force

    def handle_self_term(self):
        (cutoff, eps_rf, krf, mrf, nrf, arfm, arfn, crf) = self._get_rf_terms()

        crf_self_term = f"0.5 * ONE_4PI_EPS0 * chargeprod_ * (-crf);"
        crf_self_term += "ONE_4PI_EPS0 = {:f};".format(self.ONE_4PI_EPS0)
        crf_self_term += "crf = {:f};".format(crf)

        force_crf_self_term = openmm.CustomBondForce(crf_self_term)
        force_crf_self_term.addPerBondParameter('chargeprod_')
        force_crf_self_term.setUsesPeriodicBoundaryConditions(True)

        for i in range(self._nbf.getNumParticles()):
            ch1, _, _ = self._nbf.getParticleParameters(i)
            force_crf_self_term.addBond(i, i, [ch1*ch1])
        return force_crf_self_term

    def _get_rf_terms(self):
        cutoff, eps_rf = self._cutoff, self._eps_rf
        krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff**3)
        mrf = 4
        nrf = 6
        arfm = (3 * cutoff**(-(mrf+1))/(mrf*(nrf - mrf)))* \
            ((2*eps_rf+nrf-1)/(1+2*eps_rf))
        arfn = (3 * cutoff**(-(nrf+1))/(nrf*(mrf - nrf)))* \
            ((2*eps_rf+mrf-1)/(1+2*eps_rf))
        crf = ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff) + arfm * \
            cutoff**mrf + arfn * cutoff ** nrf
        return (cutoff, eps_rf, krf, mrf, nrf, arfm, arfn, crf)

    def _get_energy_fn(self, exception=False):
        """
        see https://github.com/rinikerlab/reeds/blob/\
        b8cf6895d08f3a85a68c892ad7d873ec129dd2c3/reeds/openmm/\
        reeds_openmm.py#L265
        """
        (cutoff, eps_rf, krf, mrf, nrf, arfm, arfn, crf) = self._get_rf_terms()

        # define additive energy terms
        #total_e = f"elec_e + lj_e;"
        total_e = "lj_e + elec_e;"
        # total_e += "elec_e = ONE_4PI_EPS0*chargeprod*(1/r + krf*r2 + arfm*r4 + arfn*r6 - crf);"
        total_e += f"elec_e = ONE_4PI_EPS0*( chargeprod*(1/r) + chargeprod_*(krf*r2 + arfm*r4 + arfn*r6 - crf));"
        total_e += f"lj_e = 4*epsilon*(sigma_over_r12 - sigma_over_r6);"
        total_e += "krf = {:f};".format(krf)
        total_e += "crf = {:f};".format(crf)
        total_e += "r6 = r2*r4;"
        total_e += "r4 = r2*r2;"
        total_e += "r2 = r*r;"
        total_e += "arfm = {:f};".format(arfm)
        total_e += "arfn = {:f};".format(arfn)
        total_e += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
        total_e += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
        total_e += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
        total_e += "sigma_over_r = sigma/r;"
        if not exception:
            total_e += "epsilon = sqrt(epsilon1*epsilon2);"
            total_e += "sigma = 0.5*(sigma1+sigma2);"
            total_e += "chargeprod = charge1*charge2;"
        total_e += "ONE_4PI_EPS0 = {:f};".format(self.ONE_4PI_EPS0)
        return total_e

def sort_indices_to_str(indices: Iterable[int]) -> str:
  sorted_indices = sorted(indices)
  return '.'.join([str(_q) for _q in sorted_indices])

def make_exception_dict(old_nbf: openmm.NonbondedForce,
  new_nbf: openmm.NonbondedForce, old_to_hybrid_map: Dict[int, int],
  new_to_hybrid_map: Dict[int, int],
  **unused_kwargs) -> Dict[openmm.Force, Dict[str, int]]:
  """retrieve a dictionary of sorted/stringed hybrid indices: original exception
  index for each force (old and new); it is generally implied that there is a
  _single_ exception between any two particles.
  """
  exception_data = {}
  for orig_force in [old_nbf, new_nbf]:
    exception_data[orig_force] = {}
    num_exceptions = orig_force.getNumExceptions()
    to_hybrid_map = old_to_hybrid_map if orig_force == old_nbf \
      else new_to_hybrid_map
    for orig_exception_idx in range(num_exceptions):
      _params = orig_force.getExceptionParameters(orig_exception_idx)
      orig_indices = _params[:2]
      hybrid_indices = [to_hybrid_map[_q] for _q in orig_indices]
      sorted_hybrid_inds_str = sort_indices_to_str(hybrid_indices)
      exception_data[orig_force][sorted_hybrid_inds_str] = orig_exception_idx
  return exception_data

def make_constraint_dict(old_system: openmm.System,
  new_system: openmm.System, old_to_hybrid_map: Dict[int, int],
  new_to_hybrid_map: Dict[int, int],
  **unused_kwargs) -> Dict[openmm.System, Dict[str, int]]:
  """
  retrieve a dictionary of sorted/stringed hybrid_indices: original constraints
  index for each system (old/new);
  """
  constraint_data = {}
  for orig_sys in [old_system, new_system]:
    constraint_data[orig_sys] = {}
    num_constraints = orig_sys.getNumConstraints()
    to_hybrid_map = old_to_hybrid_map if orig_sys == old_system \
      else new_to_hybrid_map
    for orig_const_idx in range(num_constraints):
      _params = orig_sys.getConstraintParameters(orig_const_idx)
      orig_indices = _params[:2]
      hybrid_indices = [to_hybrid_map[_q] for _q in orig_indices]
      sorted_hybrid_inds_str = sort_indices_to_str(hybrid_indices)
      constraint_data[orig_sys][sorted_hybrid_inds_str] = orig_const_idx
  return constraint_data

class SingleTopologyHybridNBFReactionFieldConverter():
  NB_PAIR_TEMPLATE = ['lj_e * step({r_cut} - reff_lj) + elec_e;',
    "elec_e = elec_term1 + elec_term2;",
    "elec_term1 = {ONE_4PI_EPS0}*chargeprod*(1/reff_q)*step({r_cut} - reff_q);",
    "elec_term2 = {ONE_4PI_EPS0}*chargeprod_*({krf}*r^2 + {arfm}*r^4 + {arfn}*r^6 - {crf});",
    "lj_e = 4*epsilon*lj_x*(lj_x-1);",
    'lj_x = (sigma/reff_lj)^6;',
    'reff_lj = sigma*((softcore_alpha*(1-lam_sub)^softcore_b + (r/sigma)^softcore_c))^(1/softcore_c);',
    'reff_q = sigma*((softcore_beta*(1-lam_sub)^softcore_e + (r/sigma)^softcore_f))^(1/softcore_f);']
  NB_SELF_TEMPLATE = "0.5*{ONE_4PI_EPS0} * chargeprod_ * (-{crf});"
  NB_GLOBAL_PARAMETERS = {
  # turn values of 1. into 1+1e-3 because of omm bug:
  # https://github.com/openmm/openmm/issues/3833
    'lambda_global': 0.,
    'softcore_alpha': 0.5,
    'softcore_beta': 0.5,
    'softcore_b': 1.001,
    'softcore_c': 6.,
    'softcore_d': 1.001,
    'softcore_e': 1.001,
    'softcore_f': 2.,
    }
  def __init__(self: Any,
    old_nbf: openmm.NonbondedForce,
    new_nbf: openmm.NonbondedForce,
    old_to_hybrid_map: Dict[int, int],
    new_to_hybrid_map: Dict[int, int],
    num_hybrid_particles: int,
    unique_old_atoms: Iterable[int],
    unique_new_atoms: Iterable[int],
    constraints_dict: Dict[openmm.System, Dict[str, int]]=None,
    cutoff: float=1.2,
    eps_rf: float=78.5,
    ONE_4PI_EPS0: float=138.93545764438198,
    allow_false_unique_exceptions: bool=True,
    **kwargs):
      self._old_nbf = old_nbf
      self._new_nbf = new_nbf
      self._old_to_hybrid_map = old_to_hybrid_map
      self._new_to_hybrid_map = new_to_hybrid_map
      self._num_hybrid_particles = num_hybrid_particles
      self._unique_old_atoms = unique_old_atoms
      self._unique_new_atoms = unique_new_atoms
      self._cutoff = cutoff
      self._eps_rf = eps_rf
      self.ONE_4PI_EPS0 = ONE_4PI_EPS0
      self._allow_false_unique_exceptions = allow_false_unique_exceptions
      self._constraints_dict = constraints_dict

      self._hybrid_to_old_map = {val:key for key, val in self._old_to_hybrid_map.items()}
      self._hybrid_to_new_map = {val:key for key, val in self._new_to_hybrid_map.items()}

      self._custom_nbf = self.handle_nonbonded_pairs(**kwargs)
      self._custom_bfs = self.make_exceptions(**kwargs)
      self._self_bf = self.make_self_force(**kwargs)

  @property
  def rf_forces(self):
    outs = []
    outs.append(copy.deepcopy(self._custom_nbf))
    outs.append(copy.deepcopy(self._custom_bfs[0]))
    outs.append(copy.deepcopy(self._custom_bfs[1]))
    outs.append(copy.deepcopy(self._self_bf))
    return outs

  def _get_aux_pair_terms(self, **kwargs):
    # this is starting to look at lot like `_get_aux_mapped_exception_terms`;
    # maybe we can solve this?
    aux_template = [
      'lam_sub = select(1-lift_somewhere, 1., lifting_selector);', # dont lift at both
      'lift_somewhere = step(lift_at_zero + lift_at_one - 0.1);'
      'lifting_selector = select(1-lift_at_zero, 1-lambda_global, lambda_global);'] # this is right!
    for term in ['chargeprod_', 'chargeprod', 'sigma', 'epsilon']:
      aux_template.append(
      f"{term} = old_{term} + lambda_global*(new_{term} - old_{term});")
    aux_template += [
       'lift_at_zero = select(abs(old_chargeprod)+abs(old_epsilon), 0, 1);',
       'lift_at_one = select(abs(new_chargeprod)+abs(new_epsilon), 0, 1);',
       'old_chargeprod_ = old_chargeprod;',
       'new_chargeprod_ = new_chargeprod;'
       'old_chargeprod = old_charge1*old_charge2;',
       'new_chargeprod = new_charge1*new_charge2;',
       'old_sigma = (old_sigma1+old_sigma2)/2;',
       'new_sigma = (new_sigma1+new_sigma2)/2;',
       'old_epsilon = sqrt(old_epsilon1*old_epsilon2);',
       'new_epsilon = sqrt(new_epsilon1*new_epsilon2);']
    perParticleParameters = ['old_charge', 'new_charge', 'old_sigma',
      'new_sigma', 'old_epsilon', 'new_epsilon']
    global_parameters = {} # we already have lambda global
    return aux_template, perParticleParameters, global_parameters

  def _get_aux_unique_exception_terms(self, **kwargs):
    """exception template for unique pairs"""
    aux_template = [
      'lam_sub = 1;', # never softcore the chargeprod interaction
      # (exc) chargeprod is retained if `retain_exception_switch`, else scaled.
      'chargeprod = select(1 - retain_exception_switch, lam_scale*in_chargeprod, in_chargeprod);',
      # (nonexc) chargeprod is always linearly interp from old to new
      'chargeprod_ = lam_scale*in_chargeprod_;',
      'sigma = in_sigma;', # sigma never changes.
      'epsilon = select(1 - retain_exception_switch, lam_scale*in_epsilon, in_epsilon);',
      # lambda scale goes from 1->0 if unique old else 0->1
      'lam_scale = select(1-unique_old, lambda_global, 1-lambda_global);']

    perBondParameters = ['in_chargeprod', 'in_chargeprod_',
      'in_sigma', 'in_epsilon', 'unique_old']
    global_parameters = {'retain_exception_switch': 1} # `lambda_global`
    return aux_template, perBondParameters, global_parameters

  def _get_aux_mapped_exception_terms(self, **kwargs):
    """exception template for mapped pairs"""
    aux_template = [
      'chargeprod = old_chargeprod + lambda_global*(new_chargeprod - old_chargeprod);',
      'chargeprod_ = old_chargeprod_ + lambda_global*(new_chargeprod_ - old_chargeprod_);',
      'epsilon = old_epsilon + lambda_global*(new_epsilon - old_epsilon);',
      'sigma = old_sigma + lambda_global*(new_sigma - old_sigma);'
      # do not lift at zero _and_ 1; typically lift at endstate if term disappears/appears
      # if term disappears, lift at 1; if term appears, lift at zero.
      'lam_sub = select(1-lift_somewhere, 1., lifting_selector);',
      'lift_somewhere = step(lift_at_zero + lift_at_one - 0.1);',
      'lifting_selector = select(1-lift_at_zero, 1-lambda_global, lambda_global);'
    ]
    perBondParameters = ['old_chargeprod', 'new_chargeprod', 'old_chargeprod_', 'new_chargeprod_',
      'old_sigma', 'new_sigma', 'old_epsilon', 'new_epsilon' ,'lift_at_zero', 'lift_at_one']
    global_parameters = {} # lambda_global is already here.
    return aux_template, perBondParameters, global_parameters

  def _get_aux_self_terms(self, **kwargs):
    """get the self term"""
    aux_template = ["0.5*{ONE_4PI_EPS0}*chargeprod_*(-{crf});"
      "chargeprod_ = old_chargeprod_ + lambda_global*(new_chargeprod_ - old_chargeprod_);"]
    perBondParameters = ['old_chargeprod_', 'new_chargeprod_']
    global_parameters = {}
    return aux_template, perBondParameters, global_parameters


  def _get_rf_terms(self, **unused_kwargs):
    """return a dict of reaction field terms."""
    cutoff, eps_rf = self._cutoff, self._eps_rf
    krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff**3)
    mrf = 4
    nrf = 6
    arfm = (3 * cutoff**(-(mrf+1))/(mrf*(nrf - mrf)))* \
        ((2*eps_rf+nrf-1)/(1+2*eps_rf))
    arfn = (3 * cutoff**(-(nrf+1))/(nrf*(mrf - nrf)))* \
        ((2*eps_rf+mrf-1)/(1+2*eps_rf))
    crf = ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff) + arfm * \
        cutoff**mrf + arfn * cutoff ** nrf
    out_dict = {'krf': krf, 'mrf': mrf, 'nrf': nrf,
      'arfm': arfm, 'arfn': arfn, 'crf': crf,
      'ONE_4PI_EPS0': self.ONE_4PI_EPS0, 'r_cut': self._cutoff}
    return out_dict

  def handle_nonbonded_pairs(self, **kwargs):
    """generate the CustomNonbondedForce with appropriate exclusions; v2"""
    nbf = openmm.CustomNonbondedForce('') # space filler for expression
    nbf.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    nbf.setCutoffDistance(self._cutoff)
    nbf.setUseLongRangeCorrection(False) # do I want to hardcode this?

    aux_template, perParticleParameters, global_parameters = self._get_aux_pair_terms(**kwargs)
    for particle_param in perParticleParameters: # add per particle params
      nbf.addPerParticleParameter(particle_param)
    all_global_params = copy.deepcopy(self.NB_GLOBAL_PARAMETERS)
    all_global_params.update(global_parameters)
    for gp_name, gp_val in all_global_params.items(): # add global params
        nbf.addGlobalParameter(gp_name, gp_val)

    hybrid_to_old_map = self._hybrid_to_old_map # for references
    hybrid_to_new_map = self._hybrid_to_new_map

    for hybrid_idx in range(self._num_hybrid_particles):
      # ['old_charge', 'new_charge', 'old_sigma',
      # 'new_sigma', 'old_epsilon', 'new_epsilon', 'unique_old']
      try: # get old idx
        old_idx = hybrid_to_old_map[hybrid_idx]
      except Exception as e:
        old_idx = -1
      try: # get new idx
        new_idx = hybrid_to_new_map[hybrid_idx]
      except Exception as e:
        new_idx = -1
      if new_idx < 0: # then it is unique old
        oc, os, oe = self._old_nbf.getParticleParameters(old_idx)
        nc, ns, ne = oc*0., os, oe*0. # new charge, eps are zero
      elif old_idx < 0: # then it is unique new
        nc, ns, ne = self._new_nbf.getParticleParameters(new_idx)
        oc, os, oe = nc*0., ns, ne*0. # old charge, eps are zero
      else: # it is mapped; there is more complicated lifting logic
        oc, os, oe = maybe_params_as_unitless(self._old_nbf.getParticleParameters(old_idx))
        nc, ns, ne = maybe_params_as_unitless(self._new_nbf.getParticleParameters(new_idx))
      _params = [oc, nc, os, ns, oe, ne]
      pass_hybrid_idx = nbf.addParticle(_params)
      assert pass_hybrid_idx == hybrid_idx

    # add exclusions between unique new/old
    for unique_new_idx in self._unique_new_atoms:
      unique_new_hybr_idx = self._new_to_hybrid_map[unique_new_idx]
      for unique_old_idx in self._unique_old_atoms:
        unique_old_hybr_idx = self._old_to_hybrid_map[unique_old_idx]
        excl_idx = nbf.addExclusion(unique_new_hybr_idx, unique_old_hybr_idx)

    # add exclusions based on existing exceptions;
    # track the hybrid exceptions that have already been added, too,
    # in order to avoid redundancies.
    excluded_particles_dict = {i: [] for i in range(self._num_hybrid_particles)}
    for orig_nbf in [self._old_nbf, self._new_nbf]:
      num_exceptions = orig_nbf.getNumExceptions()
      querier_map = self._old_to_hybrid_map if orig_nbf == self._old_nbf \
        else self._new_to_hybrid_map
      for _idx in range(num_exceptions):
        orig_p1, orig_p2, cp, s, e = orig_nbf.getExceptionParameters(_idx)
        hybr_p1, hybr_p2 = querier_map[orig_p1], querier_map[orig_p2]
        sorted_hybrid_indices = sorted([hybr_p1, hybr_p2])
        ref_idx_list = excluded_particles_dict[sorted_hybrid_indices[0]]
        if sorted_hybrid_indices[1] in ref_idx_list:# this exclusion already exists
          pass
        else: # add the exclusion and update the dict
          excl_idx = nbf.addExclusion(*sorted_hybrid_indices)
          excluded_particles_dict[sorted_hybrid_indices[0]].append(sorted_hybrid_indices[1])

    # now fix the energy expression
    rf_terms = self._get_rf_terms(**kwargs)
    full_energy_expr = ' '.join(self.NB_PAIR_TEMPLATE +
      aux_template).format(**rf_terms)
    nbf.setEnergyFunction(full_energy_expr)
    return nbf

  def make_custom_bond_forces(self, **unused_kwargs):
    """format `openmm.CustomBondForce` for exceptions"""
    out_forces = []
    for method in [self._get_aux_unique_exception_terms,
      self._get_aux_mapped_exception_terms]:
      cbf = openmm.CustomBondForce('') # blank initial energy fn
      aux_template, perBondParameters, global_parameters = method(**unused_kwargs)
      all_global_params = copy.deepcopy(self.NB_GLOBAL_PARAMETERS)
      all_global_params.update(global_parameters)
      for _param in perBondParameters:
        cbf.addPerBondParameter(_param)
      for gp_name, gp_val in all_global_params.items(): # add global params
        cbf.addGlobalParameter(gp_name, gp_val)
      rf_terms = self._get_rf_terms(**unused_kwargs)
      full_energy_expr = ' '.join(self.NB_PAIR_TEMPLATE + aux_template).format(**rf_terms)
      cbf.setEnergyFunction(full_energy_expr)
      out_forces.append(cbf)
    return out_forces

  def make_exceptions(self, **unused_kwargs):
    """make the unique and mapped exception `openmm.CustomBondForce`
    objects to accommodate nonbonded exclusions in the
    `openmm.CustomNonbondedForce`.
    TODO: we also want to appropriately handle constraints properly here;
        perhaps it should go into the self force since they are all
        independent of r (set by immutable parameter r0).
    """
    unique_bf, custom_bf = self.make_custom_bond_forces(**unused_kwargs)
    hybrid_to_old_map = self._hybrid_to_old_map
    hybrid_to_new_map = self._hybrid_to_new_map

    # first, get the exception data to query later.
    exception_data = make_exception_dict(old_nbf = self._old_nbf,
      new_nbf = self._new_nbf, old_to_hybrid_map = self._old_to_hybrid_map,
      new_to_hybrid_map = self._new_to_hybrid_map, **unused_kwargs)

    # now iterate over old/new nbf
    for _force in [self._old_nbf, self._new_nbf]:
      num_exceptions = _force.getNumExceptions() # query num exceptions
      opp_force = self._old_nbf if _force == self._new_nbf else self._new_nbf
      to_hybrid_map = self._old_to_hybrid_map if _force == self._old_nbf \
        else self._new_to_hybrid_map
      to_opposite_orig_map = hybrid_to_new_map if _force == self._old_nbf \
        else hybrid_to_old_map
      opp_exc_dict_to_query = exception_data[self._new_nbf] if \
        _force == self._old_nbf else exception_data[self._old_nbf]
      uniques = self._unique_old_atoms if _force == self._old_nbf \
        else self._unique_new_atoms
      for orig_exc_idx in range(num_exceptions): # query original exceptions
        orig_exc_params = _force.getExceptionParameters(orig_exc_idx)
        orig_indices = orig_exc_params[:2]
        orig_nonidx_params = maybe_params_as_unitless(orig_exc_params[2:])
        hybrid_inds = [to_hybrid_map[_q] for _q in orig_indices]
        sorted_hybrid_inds_str = sort_indices_to_str(hybrid_inds)
        # query charges, since we need these anyway
        c1, _, _ = _force.getParticleParameters(orig_indices[0])
        c2, _, _ = _force.getParticleParameters(orig_indices[1])
        contains_unique = len(
          set(orig_indices).intersection(set(uniques))) > 0
        if contains_unique: # place this into the unique CustomBondForce
          # ['in_chargeprod', 'in_chargeprod_', 'in_sigma', 'in_epsilon', 'unique_old']
          is_uold = _force == self._old_nbf
          _params = [orig_nonidx_params[0], c1*c2, orig_nonidx_params[1],
          orig_nonidx_params[2], is_uold]
          _ = unique_bf.addBond(*hybrid_inds, _params)
          continue # now all uniques are handled, we can skip the rest

        # always query the opposite particle indices' (charges)
        opp_particle_indices = [to_opposite_orig_map[_q] for
          _q in hybrid_inds]

        opp_c1, _, _ = opp_force.getParticleParameters(opp_particle_indices[0])
        opp_c2, _, _ = opp_force.getParticleParameters(opp_particle_indices[1])
        try: # get the exception from the opposite system
          opp_exc_idx = opp_exc_dict_to_query[sorted_hybrid_inds_str]
        except Exception as e: # the exception idx doesn't exist;
          opp_exc_idx = -1
        if opp_exc_idx > -1: # then this exception is mapped appropriately
          opposite_parameters = maybe_params_as_unitless(
            opp_force.getExceptionParameters(opp_exc_idx)[2:])
          if _force == self._old_nbf: # only do this for the old force
            # since adding new force terms would be redundant
            # ['old_chargeprod', 'new_chargeprod', 'old_chargeprod_', 'new_chargeprod_',
            # 'old_sigma', 'new_sigma', 'old_epsilon', 'new_epsilon' ,'lift_at_zero', 'lift_at_one']
            old_chargeprod, new_chargeprod = orig_nonidx_params[0], opposite_parameters[0]
            old_chargeprod_, new_chargeprod_ = c1*c2, opp_c1*opp_c2
            old_sigma, new_sigma = orig_nonidx_params[1], opposite_parameters[1]
            old_epsilon, new_epsilon = orig_nonidx_params[2], opposite_parameters[2]
            # revisit this again...
            lift_at_zero = 0 if (abs(old_chargeprod)>0 or abs(old_epsilon)>0) else 1
            lift_at_one = 0 if (abs(new_chargeprod)>0 or abs(new_epsilon)>0) else 1
            _params = [old_chargeprod, new_chargeprod, old_chargeprod_, new_chargeprod_,
              old_sigma, new_sigma, old_epsilon, new_epsilon, lift_at_zero, lift_at_one]
            _ = custom_bf.addBond(*hybrid_inds, _params)
          else: # iterating over new nbf is already handled above
            pass
        else: # then the exception exists in one force but not the other.
          if self._allow_false_unique_exceptions:
            here_old_nbf = _force == self._old_nbf
            # this is the tricky bit, as we may need to softcore this interaction
            old_chargeprod = orig_nonidx_params[0] if here_old_nbf else 0.
            new_chargeprod = orig_nonidx_params[0] if not here_old_nbf else 0.
            old_chargeprod_ = c1*c2 if here_old_nbf else opp_c1*opp_c2
            new_chargeprod_ = c1*c2 if not here_old_nbf else opp_c1*opp_c2
            # keep sigma constant for stability
            old_sigma, new_sigma = orig_nonidx_params[1], orig_nonidx_params[1]
            old_epsilon = orig_nonidx_params[2] if here_old_nbf else 0.
            new_epsilon = orig_nonidx_params[2] if not here_old_nbf else 0.
            # always lift at the opposite endstate.
            lift_at_zero = 0. if here_old_nbf else 1.
            lift_at_one = 0. if not here_old_nbf else 1.
            _params = [old_chargeprod, new_chargeprod, old_chargeprod_, new_chargeprod_,
                old_sigma, new_sigma, old_epsilon, new_epsilon, lift_at_zero, lift_at_one]
            _ = custom_bf.addBond(*hybrid_inds, _params)
          else:
            raise Exception(f"""disallowing false unique exception parameterization;
              this bit of code is bleeding edge and not well-tested.
              revisit your mapping strategy, please.""")
        # and that's all the exceptions.
    return unique_bf, custom_bf

  def make_self_force(self, **unused_kwargs):
    bf = openmm.CustomBondForce('') # space filler for expression
    rf_terms = self._get_rf_terms(**unused_kwargs)
    bf.setUsesPeriodicBoundaryConditions(True)

    aux_template, perBondParameters, global_parameters = self._get_aux_self_terms(**unused_kwargs)
    for bond_param in perBondParameters: # add per particle params
      bf.addPerBondParameter(bond_param)
    all_global_params = copy.deepcopy(self.NB_GLOBAL_PARAMETERS)
    all_global_params.update(global_parameters)
    for gp_name, gp_val in all_global_params.items(): # add global params
      bf.addGlobalParameter(gp_name, gp_val)
    energy_fn = ' '.join(aux_template).format(**rf_terms)
    bf.setEnergyFunction(energy_fn)

    # now add the terms
    for i in range(self._num_hybrid_particles):
      try:
        old_c1, _, _ = self._old_nbf.getParticleParameters(
          self._hybrid_to_old_map[i])
      except: # it's unique new
        old_c1 = 0.
      try:
        new_c1, _, _ = self._new_nbf.getParticleParameters(
          self._hybrid_to_new_map[i])
      except: # it's unique old
        new_c1 = 0.
      _ = bf.addBond(i, i, [old_c1*old_c1, new_c1*new_c1])
    return bf
