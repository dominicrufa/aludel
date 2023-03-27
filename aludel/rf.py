"""reaction field conversion"""

import openmm
from openmm import app, unit
import numpy as np
import copy
from typing import Any, Tuple, Dict, List, Callable, Set
from aludel.utils import maybe_params_as_unitless
from copy import deepcopy


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
                 system: openmm.System,
                 cutoff: float = 1.2,
                 eps_rf: float = 78.5,
                 ONE_4PI_EPS0: float = 138.93545764438198,
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
        self._is_periodic = self._nbf.usesPeriodicBoundaryConditions()  # whether it uses pbcs...
        self._system = system
        self._cutoff = cutoff if self._is_periodic else 99.
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
        custom_nb_force.addPerParticleParameter('charge')  # always add

        custom_nb_force.addPerParticleParameter('sigma')
        custom_nb_force.addPerParticleParameter('epsilon')

        nb_method = openmm.CustomNonbondedForce.CutoffPeriodic if \
            self._is_periodic else openmm.CustomNonbondedForce.NoCutoff
        custom_nb_force.setNonbondedMethod(nb_method)
        custom_nb_force.setCutoffDistance(self._cutoff)
        custom_nb_force.setUseLongRangeCorrection(False)  # for lj, never

        # add particles
        for idx in range(self._nbf.getNumParticles()):
            c, s, e = self._nbf.getParticleParameters(idx)
            custom_nb_force.addParticle([c, s, e])

        # add exclusions from nbf exceptions
        for idx in range(self._nbf.getNumExceptions()):
            j, k, _, _, _ = self._nbf.getExceptionParameters(idx)
            custom_nb_force.addExclusion(j, k)
        return custom_nb_force

    def handle_nb_exceptions(self):
        energy_fn = self._get_energy_fn(exception=True)
        custom_b_force = openmm.CustomBondForce(energy_fn)
        # add terms separately so we need not reimplement the energy fn
        for _param in ['chargeprod', 'sigma', 'epsilon', 'chargeprod_']:
            custom_b_force.addPerBondParameter(_param)

        # copy exceptions
        for idx in range(self._nbf.getNumExceptions()):
            j, k, chargeprod, mix_sigma, mix_epsilon = self._nbf. \
                getExceptionParameters(idx)

            # now query charges, sigma, epsilon
            c1, _, _ = self._nbf.getParticleParameters(j)
            c2, _, _ = self._nbf.getParticleParameters(k)

            custom_b_force.addBond(j, k,
                                   [chargeprod, mix_sigma, mix_epsilon, c1 * c2])

        return custom_b_force

    def handle_self_term(self):
        (cutoff, eps_rf, krf, mrf, nrf, arfm, arfn, crf) = self._get_rf_terms()

        crf_self_term = f"0.5 * ONE_4PI_EPS0 * chargeprod_ * (-crf);"
        crf_self_term += "ONE_4PI_EPS0 = {:f};".format(self.ONE_4PI_EPS0)
        crf_self_term += "crf = {:f};".format(crf)

        force_crf_self_term = openmm.CustomBondForce(crf_self_term)
        force_crf_self_term.addPerBondParameter('chargeprod_')

        force_crf_self_term.setUsesPeriodicBoundaryConditions(self._is_periodic)

        for i in range(self._nbf.getNumParticles()):
            ch1, _, _ = self._nbf.getParticleParameters(i)
            force_crf_self_term.addBond(i, i, [ch1 * ch1])
        return force_crf_self_term

    def _get_rf_terms(self):
        cutoff, eps_rf = self._cutoff, self._eps_rf
        krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff ** 3)
        mrf = 4
        nrf = 6
        arfm = (3 * cutoff ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * \
               ((2 * eps_rf + nrf - 1) / (1 + 2 * eps_rf))
        arfn = (3 * cutoff ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * \
               ((2 * eps_rf + mrf - 1) / (1 + 2 * eps_rf))
        crf = ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff) + arfm * \
              cutoff ** mrf + arfn * cutoff ** nrf
        return (cutoff, eps_rf, krf, mrf, nrf, arfm, arfn, crf)

    def _get_energy_fn(self, exception=False):
        """
        see https://github.com/rinikerlab/reeds/blob/\
        b8cf6895d08f3a85a68c892ad7d873ec129dd2c3/reeds/openmm/\
        reeds_openmm.py#L265
        """
        (cutoff, eps_rf, krf, mrf, nrf, arfm, arfn, crf) = self._get_rf_terms()

        # define additive energy terms
        # total_e = f"elec_e + lj_e;"
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


# Utility functions for HybridReactionFieldConverter

def sort_indices_to_str(indices: List[int]) -> str:
    sorted_indices = sorted(indices)
    return '.'.join([str(_q) for _q in sorted_indices])


def make_exception_dict(
        old_nbf: openmm.NonbondedForce,
        new_nbf: openmm.NonbondedForce, old_to_hybrid_map: Dict[int, int],
        new_to_hybrid_map: Dict[int, int], **unused_kwargs) -> Dict[openmm.Force, Dict[str, int]]:
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
                         new_to_hybrid_map: Dict[int, int], **unused_kwargs) -> Dict[openmm.System, Dict[str, int]]:
    """retrieve a dictionary of sorted/stringed hybrid_indices: original constraints
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


def are_nb_params_identical(oc, nc, os, ns, oe, ne, oc_=None, nc_=None, **unused_kwargs):
    """take a set of old/new charge, sigma, epsilon; return if they change"""
    if oc_ is None: assert nc_ is None, f"both aux c_s must be None if one is"
    charge_match = np.isclose(oc, nc)
    sigma_match = np.isclose(os, ns)
    eps_match = np.isclose(oe, ne)
    charge_und_match = True if oc_ is None else np.isclose(oc_, nc_)

    if not charge_match:  # if the charges do not match, fail
        return False
    if not charge_und_match:
        return False
    if not eps_match:  # if epsilons do not match, fail
        return False
    if not sigma_match:  # epsilons are identical; if sigmas do not match, fail
        return False
    return True


class SingleTopologyHybridNBFReactionFieldConverter():
    """In this `Converter` object, I am modifying the typical nonbonded and exception forces in-place.
    Each force will be duplicated. For each force object, there will be one force that is absolutely immutable w.r.t.
    `lambda_global` and one force that is tethered so that we can get a speedup and
    only have to re-evaluate certain force objects.

    NOTE: there should not be omissions to this object because 1-4 terms are _always_ interpolated, unlike valence force,
    of which, terms containing unique particles are typically retained (as is typical of RBFEs)
    """

    NB_PAIR_TEMPLATE = ['lj_e * step({r_cut} - reff_lj) + elec_e;',
                        "elec_e = elec_term1 + elec_term2;",
                        "elec_term1 = {ONE_4PI_EPS0}*res_charge*(1/reff_q)*step({r_cut} - reff_q);",
                        "elec_term2 = {ONE_4PI_EPS0}*res_charge_*({krf}*r^2 + {arfm}*r^4 + {arfn}*r^6 - {crf});",
                        "lj_e = 4*res_epsilon*lj_x*(lj_x-1);",
                        'lj_x = (res_sigma/reff_lj)^6;']  # define the general nonbonded pair template
    NB_MIXING_PARAMETERS = [
        'res_charge=res_charge1 * res_charge2;'
        'res_charge_=res_charge1 * res_charge2;',
        'res_sigma=(res_sigma1 + res_sigma2)/2;',
        'res_epsilon=sqrt(res_epsilon1 * res_epsilon2);']  # generic per-particle parameters

    NB_PER_PARTICLE_PARAMETERS = ['res_charge', 'res_sigma', 'res_epsilon']
    NB_STANDARD_REFF_TEMPLATE = ['reff_lj = r;', 'reff_q = r;']  # define the standard r-effectives

    NB_ALCH_REFF_TEMPLATE = [  # define the softcored r-effectives
        'reff_lj = res_sigma*(({softcore_alpha}*(lam_sub)^{softcore_b} + (r/res_sigma)^{softcore_c}))^(1/{softcore_c});',
        'reff_q = res_sigma*(({softcore_beta}*(lam_sub)^{softcore_e} + (r/res_sigma)^{softcore_f}))^(1/{softcore_f});']
    NB_ALCH_LIFTING_SELECTOR = [
        'lam_sub = (unique_old * lambda_global) + (unique_new * (1. - lambda_global));',
        'unique_old = step(unique_old1 + unique_old2 - 0.1);',
        'unique_new = step(unique_new1 + unique_new2 - 0.1);']
    NB_ALCH_MIXING_PARAMETERS = [
        'res_charge_ = res_charge;',  # same as `res_charge`
        'res_charge = old_charge + lambda_global * (new_charge - old_charge);',
        'res_sigma = old_sigma + lambda_global * (new_sigma - old_sigma);',
        'res_epsilon = old_epsilon + lambda_global * (new_epsilon - old_epsilon);'
    ]

    NB_ALCH_OLD_NEW_MIXING_PARAMETERS = [
        'old_charge = old_charge1 * old_charge2;',
        'new_charge = new_charge1 * new_charge2;',
        'old_sigma = (old_sigma1 + old_sigma2)/2;',
        'new_sigma = (new_sigma1 + new_sigma2)/2;',
        'old_epsilon = sqrt(old_epsilon1 * old_epsilon2);',
        'new_epsilon = sqrt(new_epsilon1 * new_epsilon2);'
    ]
    NB_ALCH_PER_PARTICLE_PARAMETERS = [
        'old_charge', 'new_charge',
        'old_sigma', 'new_sigma',
        'old_epsilon', 'new_epsilon',
        'unique_old', 'unique_new']

    NB_EXC_ALCH_MIXING_PARAMETERS = deepcopy(NB_ALCH_MIXING_PARAMETERS)
    NB_EXC_ALCH_MIXING_PARAMETERS[0] = 'res_charge_ = old_charge_ + lambda_global * (new_charge_ - old_charge_);'
    NB_EXC_STANDARD_PER_BOND_PARAMETERS = deepcopy(NB_PER_PARTICLE_PARAMETERS) + ['res_charge_']

    NB_SELF_TEMPLATE = "0.5*{ONE_4PI_EPS0} * chargeprod_ * (-{crf});"  # define the nonbonded self term template
    NB_GLOBAL_PARAMETERS = {  # define the global parameters
        # turn values of 1. into 1+1e-3 because of omm bug:
        # https://github.com/openmm/openmm/issues/3833
        'lambda_global': 0.}
    NB_SOFTCORE_GLOBAL_PARAMETERS = {
        'softcore_alpha': 0.5,
        'softcore_beta': 0.5,
        'softcore_b': 1.001,
        'softcore_c': 6.,
        'softcore_d': 1.001,
        'softcore_e': 1.001,
        'softcore_f': 2.
    }

    def __init__(self: Any,
                 old_nbf: openmm.NonbondedForce,
                 new_nbf: openmm.NonbondedForce,
                 old_to_hybrid_map: Dict[int, int],
                 new_to_hybrid_map: Dict[int, int],
                 num_hybrid_particles: int,
                 unique_old_atoms: List[int],
                 unique_new_atoms: List[int],
                 constraints_dict: Dict[openmm.System, Dict[Tuple[int], float]] = {},
                 cutoff: float = 1.2,
                 eps_rf: float = 78.5,
                 ONE_4PI_EPS0: float = 138.93545764438198,
                 allow_false_unique_exceptions: bool = True,
                 **kwargs):

        self._consistent_mapped_params = None
        self._inconsistent_mapped_indices = None
        self._unique_exception_indices = None
        self._alch_interaction_group = None

        self._old_nbf = old_nbf
        self._new_nbf = new_nbf
        self._is_periodic = self._old_nbf.usesPeriodicBoundaryConditions()
        self._old_to_hybrid_map = old_to_hybrid_map
        self._new_to_hybrid_map = new_to_hybrid_map
        self._num_hybrid_particles = num_hybrid_particles
        self._unique_old_atoms = unique_old_atoms
        self._unique_new_atoms = unique_new_atoms
        self._cutoff = cutoff if self._is_periodic else 99.
        self._eps_rf = eps_rf
        self.ONE_4PI_EPS0 = ONE_4PI_EPS0
        self._allow_false_unique_exceptions = allow_false_unique_exceptions
        self._constraints_dict = constraints_dict

        self._hybrid_to_old_map = {val: key for key, val in self._old_to_hybrid_map.items()}
        self._hybrid_to_new_map = {val: key for key, val in self._new_to_hybrid_map.items()}

        self._custom_nbfs = self.handle_nonbonded_pairs(**kwargs)
        self.create_exception_dicts(**kwargs)
        self._custom_bfs = self.make_exceptions(**kwargs)
        self._self_bf = self.make_self_force(**kwargs)

    @property
    def rf_forces(self):
        """retrieve a nested dict of `Force.__class__.__name`: Dict[`type`<str>: `openmm.Force`]"""
        # custombondedforces are handled in a special way
        custom_bf_forces = copy.deepcopy(self._custom_bfs)
        custom_bf_forces.update(copy.deepcopy(self._self_bf))

        # make out dict
        out_dict = {'CustomNonbondedForce': copy.deepcopy(self._custom_nbfs),
                    'CustomBondForce': custom_bf_forces}

        return out_dict

    def create_exception_dicts(self, **kwargs):
        """log a collection of attribute dictionaries;
        1. `unique_exception_indices`: dictionary of unique exception indices in old/new nbfs
        2. `inconsistent_mapped_indices`: dictionary of inconsistent unique exceptions in old/new nbfs
        3. `consistent_mapped_params`: dictionary of consistent mapped parameters that appear in both systems;
            these are either alchemical, nonalchemical, or constrained; there is further logic downfield"""
        self._unique_exception_indices = {self._old_nbf: [], self._new_nbf: []}
        self._inconsistent_mapped_indices = {self._old_nbf: [], self._new_nbf: []}
        self._consistent_mapped_params = {}

        for orig_nbf in [self._old_nbf, self._new_nbf]:  # iterate over old/new nbfs
            num_exceptions = orig_nbf.getNumExceptions()
            opp_orig_force = self._new_nbf if orig_nbf == self._old_nbf else self._old_nbf
            querier_map = self._old_to_hybrid_map if orig_nbf == self._old_nbf \
                else self._new_to_hybrid_map
            to_opp_map = self._hybrid_to_new_map if orig_nbf == self._old_nbf else self._hybrid_to_old_map
            uniques = self._unique_old_atoms if orig_nbf == self._old_nbf else self._unique_new_atoms
            for _idx in range(num_exceptions):  # iterate over all exceptions
                orig_p1, orig_p2, cp, s, e = orig_nbf.getExceptionParameters(_idx)
                exception_params = maybe_params_as_unitless([cp, s, e])
                sorted_orig_indices = sorted([orig_p1, orig_p2])
                sorted_hybrid_indices = sorted([querier_map[orig_p1], querier_map[orig_p2]])
                query_orig_idx = sorted_orig_indices[0]
                if len(set(sorted_orig_indices).intersection(set(uniques))) > 0:  # then it is unique
                    self._unique_exception_indices[orig_nbf].append(_idx)
                else:  # both of these particles are mapped
                    sorted_opp_indices = sorted([to_opp_map[sorted_hybrid_indices[0]],
                                                 to_opp_map[sorted_hybrid_indices[1]]])
                    # ask if the term is in the opposite force
                    match_idx_list = self._excluded_particles_dict[opp_orig_force][sorted_opp_indices[0]]
                    if sorted_opp_indices[1] in match_idx_list:  # this exception lives in the opp force
                        if orig_nbf == self._new_nbf: continue  # this term is already mapped since we do old then new
                        opp_exception_params = self._exception_parameters[opp_orig_force][tuple(sorted_opp_indices)]
                        self._consistent_mapped_params[tuple(sorted_hybrid_indices)] = [exception_params, opp_exception_params]
                    else:  # this exception does not live in the opp force
                        self._inconsistent_mapped_indices[orig_nbf].append(_idx)

    def _make_consistent_exclusions(self, nbf_list, **unused_kwargs):
        """add consistent exclusions from old/new nbfs in place and create attributes for queryable exception
        parameters and exclusions """
        self._excluded_particles_dict = {
            self._old_nbf: {i: [] for i in range(self._old_nbf.getNumParticles())},
            self._new_nbf: {i: [] for i in range(self._new_nbf.getNumParticles())}
        }

        self._exception_parameters = {
            self._old_nbf: {},
            self._new_nbf: {}
        }

        internal_hybrid_exceptions = {i: [] for i in range(self._num_hybrid_particles)}

        for orig_nbf in [self._old_nbf, self._new_nbf]:
            num_exceptions = orig_nbf.getNumExceptions()
            querier_map = self._old_to_hybrid_map if orig_nbf == self._old_nbf \
                else self._new_to_hybrid_map
            for _idx in range(num_exceptions):
                orig_p1, orig_p2, cp, s, e = orig_nbf.getExceptionParameters(_idx)
                cp, s, e = maybe_params_as_unitless([cp, s, e])  # make terms unitless
                sorted_orig_indices = sorted([orig_p1, orig_p2])
                self._exception_parameters[orig_nbf][tuple(sorted_orig_indices)] = [cp, s, e]  # record the exception
                hybr_p1, hybr_p2 = querier_map[orig_p1], querier_map[orig_p2]
                sorted_hybrid_indices = sorted([hybr_p1, hybr_p2])
                ref_idx_list = internal_hybrid_exceptions[sorted_hybrid_indices[0]]
                # regardless, append to excluded particles dict
                self._excluded_particles_dict[orig_nbf][sorted_orig_indices[0]].append(sorted_orig_indices[1])
                if sorted_hybrid_indices[1] in ref_idx_list:  # this exclusion already exists
                    continue
                else:  # add the exclusion and update the dict
                    internal_hybrid_exceptions[sorted_hybrid_indices[0]].append(sorted_hybrid_indices[1])
                    for nbf in nbf_list:
                        excl_idx = nbf.addExclusion(hybr_p1, hybr_p2)

    def _make_unique_exclusions(self, nbf_list, **unused_kwargs):
        """make exclusions between unique new/old particles in place"""
        for unique_new_idx in self._unique_new_atoms:
            unique_new_hybr_idx = self._new_to_hybrid_map[unique_new_idx]
            for unique_old_idx in self._unique_old_atoms:
                unique_old_hybr_idx = self._old_to_hybrid_map[unique_old_idx]
                for nbf in nbf_list:
                    excl_idx = nbf.addExclusion(unique_new_hybr_idx, unique_old_hybr_idx)

    def handle_nbf_nb_method(self, nbf, **unused_kwargs):
        nb_method = openmm.CustomNonbondedForce.CutoffPeriodic if \
            self._is_periodic else openmm.CustomNonbondedForce.NoCutoff
        nbf.setNonbondedMethod(nb_method)
        nbf.setCutoffDistance(self._cutoff)
        nbf.setUseLongRangeCorrection(False)  # do I want to hardcode this?

    def _make_alch_nbf(self, **kwargs):
        nbf = openmm.CustomNonbondedForce('')
        self.handle_nbf_nb_method(nbf)

        rf_terms = self._get_rf_terms(**kwargs)  # get the RF terms
        rf_terms.update(self.NB_SOFTCORE_GLOBAL_PARAMETERS)
        energy_expr = ' '.join(self.NB_PAIR_TEMPLATE
                               + self.NB_ALCH_REFF_TEMPLATE + self.NB_ALCH_LIFTING_SELECTOR + self.NB_ALCH_MIXING_PARAMETERS + self.NB_ALCH_OLD_NEW_MIXING_PARAMETERS).format(
            **rf_terms)

        nbf.setEnergyFunction(energy_expr)
        for particle_param in self.NB_ALCH_PER_PARTICLE_PARAMETERS:  # per particle params
            _ = nbf.addPerParticleParameter(particle_param)
        for global_param, global_param_val in self.NB_GLOBAL_PARAMETERS.items():  # global params
            _ = nbf.addGlobalParameter(global_param, global_param_val)

        return nbf

    def _make_nonalch_nbf(self, **kwargs):
        nbf = openmm.CustomNonbondedForce('')
        self.handle_nbf_nb_method(nbf)

        rf_terms = self._get_rf_terms(**kwargs)  # get the RF terms
        rf_terms.update(self.NB_SOFTCORE_GLOBAL_PARAMETERS)
        energy_expr = ' '.join(self.NB_PAIR_TEMPLATE
                               + self.NB_STANDARD_REFF_TEMPLATE + self.NB_MIXING_PARAMETERS).format(**rf_terms)

        nbf.setEnergyFunction(energy_expr)
        for particle_param in self.NB_PER_PARTICLE_PARAMETERS:  # per particle params
            _ = nbf.addPerParticleParameter(particle_param)

        return nbf

    def handle_nonbonded_pairs(self, **kwargs):
        """generate a dict of `openmm.CustomNonbondedForce` objects.
        1. independent of all global parameters; nonbonded interactions
            that do not involve parameter changes
        2. dependent on global parameters; includes nonbonded that involve parameter changes
        """
        nonalch_nbf = self._make_nonalch_nbf(**kwargs)
        alch_nbf = self._make_alch_nbf()

        # for references
        hybrid_to_old_map = self._hybrid_to_old_map
        hybrid_to_new_map = self._hybrid_to_new_map

        alch_interaction_group = []
        nonalch_interaction_group = []

        # iterate over all hybrid particles
        for hybrid_idx in range(self._num_hybrid_particles):
            old_idx = hybrid_to_old_map.get(hybrid_idx, -1)
            new_idx = hybrid_to_new_map.get(hybrid_idx, -1)

            if new_idx == -1:  # then it is unique old
                oc, os, oe = maybe_params_as_unitless(
                    self._old_nbf.getParticleParameters(old_idx))
                nc, ns, ne = oc * 0., os, oe * 0.  # new charge, eps are zero
                uniques = [1, 0]
            elif old_idx == -1:  # then it is unique new
                nc, ns, ne = maybe_params_as_unitless(
                    self._new_nbf.getParticleParameters(new_idx))
                oc, os, oe = nc * 0., ns, ne * 0.  # old charge, eps are zero
                uniques = [0, 1]
            else:  # it is mapped; there is more complicated lifting logic
                oc, os, oe = maybe_params_as_unitless(self._old_nbf.getParticleParameters(old_idx))
                nc, ns, ne = maybe_params_as_unitless(self._new_nbf.getParticleParameters(new_idx))
                # assert (oe > 0.) and np.abs(ne > 0), f"""if the particle is mapped,
                #     its old/new epsilons must be nonzero"""
                uniques = [0, 0]

            # pull the particle parameters together
            dep_params = [oc, nc, os, ns, oe, ne] + uniques

            # now add particle.
            pass_indep_hybr_idx = nonalch_nbf.addParticle([oc, os, oe])
            pass_dep_hybr_idx = alch_nbf.addParticle(dep_params)

            # the indices must match those of hybrid index
            assert pass_indep_hybr_idx == hybrid_idx
            assert pass_dep_hybr_idx == hybrid_idx

            # interaction group logic
            if np.isclose(np.sum(dep_params[-2:]), 0):  # it is not unique
                if are_nb_params_identical(*dep_params[:-2]):  # if params are identical, no need to make alchemical
                    nonalch_interaction_group.append(hybrid_idx)
                else:  # if they are not identical, need to interpolate
                    alch_interaction_group.append(hybrid_idx)
            else:  # unique particles always go to alch interaction group
                alch_interaction_group.append(hybrid_idx)

        # add exclusions
        self._make_unique_exclusions([alch_nbf, nonalch_nbf])
        self._make_consistent_exclusions([alch_nbf, nonalch_nbf])

        nonalch_nbf.addInteractionGroup(nonalch_interaction_group, nonalch_interaction_group)
        alch_nbf.addInteractionGroup(alch_interaction_group, nonalch_interaction_group)
        alch_nbf.addInteractionGroup(alch_interaction_group, alch_interaction_group)

        # define interaction groups
        self._alch_interaction_group = alch_interaction_group
        self._nonalch_interaction_group = nonalch_interaction_group

        return {'static': nonalch_nbf, 'dynamic': alch_nbf}

    def _make_alch_cbf(self, **kwargs):
        """make an alchemical `openmm.CustomBondForce`;
        bond parameters are added like [oc, nc, os, ns, oe, ne, unique_old, unique_new, oc_, nc_]"""
        cbf = openmm.CustomBondForce('')
        rf_terms = self._get_rf_terms(**kwargs)  # get the RF terms
        rf_terms.update(self.NB_SOFTCORE_GLOBAL_PARAMETERS)
        energy_expr = ' '.join(self.NB_PAIR_TEMPLATE + self.NB_ALCH_REFF_TEMPLATE + self.NB_ALCH_LIFTING_SELECTOR[
                                                                                    :1] + self.NB_EXC_ALCH_MIXING_PARAMETERS).format(
            **rf_terms)
        cbf.setEnergyFunction(energy_expr)
        per_bond_params = deepcopy(self.NB_ALCH_PER_PARTICLE_PARAMETERS) + ['old_charge_', 'new_charge_']
        for per_bond_param in per_bond_params:
            cbf.addPerBondParameter(per_bond_param)
        for global_param, global_param_val in self.NB_GLOBAL_PARAMETERS.items():  # global params
            _ = cbf.addGlobalParameter(global_param, global_param_val)
        return cbf

    def _make_standard_cbf(self, **kwargs):
        """make a nonalchemical `openmm.CustomBondForce`;
        bond parameters are added like [c, s, e, c_]"""
        cbf = openmm.CustomBondForce('')
        rf_terms = self._get_rf_terms(**kwargs)  # get the RF terms
        energy_expr = ' '.join(self.NB_PAIR_TEMPLATE + self.NB_STANDARD_REFF_TEMPLATE).format(**rf_terms)
        cbf.setEnergyFunction(energy_expr)
        for per_bond_param in self.NB_EXC_STANDARD_PER_BOND_PARAMETERS:
            cbf.addPerBondParameter(per_bond_param)
        return cbf

    def _make_standard_constraint_cbf(self, **kwargs):
        """make a nonalchemical, constrained `openmm.CustomBondForce`;
        bond parameters are added like [c, s, e, c_, reff_q]; the last two params are generally the same"""
        cbf = openmm.CustomBondForce('')
        rf_terms = self._get_rf_terms(**kwargs)  # get the RF terms
        energy_expr = ' '.join(self.NB_PAIR_TEMPLATE).format(**rf_terms)
        energy_expr = energy_expr.replace('r^', 'reff_q^').replace('reff_lj', 'reff_q')  # make the expression independent of r
        cbf.setEnergyFunction(energy_expr)
        per_bond_params = self.NB_EXC_STANDARD_PER_BOND_PARAMETERS + ['reff_q']
        for param in per_bond_params:
            _ = cbf.addPerBondParameter(param)
        return cbf

    def _make_inconsistent_exception_force(self, **kwargs):
        """make a force that handles inconsistent exceptions; specifically, this force turns interactions that
        exist as typical pairwise interactions in the new force but not in the old force and vice versa"""
        cbf = self._make_alch_cbf(**kwargs)
        for orig_nbf, inconsistent_indices in self._inconsistent_mapped_indices.items():
            to_hybrid_map = self._old_to_hybrid_map if orig_nbf == self._old_nbf else self._new_to_hybrid_map
            for inconsistent_idx in inconsistent_indices:
                p1, p2, _, _, _ = orig_nbf.getExceptionParameters(inconsistent_idx)
                hybrid_p1, hybrid_p2 = to_hybrid_map[p1], to_hybrid_map[p2]
                [old_p1, old_p2] = [self._hybrid_to_old_map[to_hybrid_map[p1]], self._hybrid_to_old_map[to_hybrid_map[p2]]]
                [new_p1, new_p2] = [self._hybrid_to_new_map[to_hybrid_map[p1]], self._hybrid_to_new_map[to_hybrid_map[p2]]]
                oc1, os1, oe1 = maybe_params_as_unitless(self._old_nbf.getParticleParameters(old_p1))
                oc2, os2, oe2 = maybe_params_as_unitless(self._old_nbf.getParticleParameters(old_p2))
                nc1, ns1, ne1 = maybe_params_as_unitless(self._new_nbf.getParticleParameters(new_p1))
                nc2, ns2, ne2 = maybe_params_as_unitless(self._new_nbf.getParticleParameters(new_p2))
                if orig_nbf == self._old_nbf:  # we take the new parameters
                    params = [0., nc1 * nc2,
                            0.5 * (ns1 + nc2) / 2, 0.5 * (ns1 + nc2) / 2,
                            0., np.sqrt(ne1 * ne2),
                            0, 1,
                            0., nc1 * nc2]
                else:
                    params = [oc1 * oc2, 0.,
                            0.5 * (os1 + os2), 0.5 * (os1 + os2),
                            np.sqrt(oe1 * oe2), 0.,
                            1, 0,
                            oc1 * oc2, 0.]
                cbf.addBond(hybrid_p1, hybrid_p2, params)
        return {'dynamic_inconsistent': cbf}

    def _make_unique_exception_force(self, **kwargs):
        """make an exception force that handles the unique old/new particle exceptions (they are to be turned off/on);
        this is with specific reference to `self._unique_exception_indices`"""
        cbf = self._make_alch_cbf(**kwargs)
        for orig_nbf, unique_exception_indices in self._unique_exception_indices.items():
            to_hybrid_map = self._old_to_hybrid_map if orig_nbf == self._old_nbf else self._new_to_hybrid_map
            for unique_idx in unique_exception_indices:
                p1, p2, c, s, e = orig_nbf.getExceptionParameters(unique_idx)
                hybrid_p1, hybrid_p2 = to_hybrid_map[p1], to_hybrid_map[p2]
                c, s, e = self._exception_parameters[orig_nbf][tuple(sorted([p1, p2]))]
                c1, _, _ = maybe_params_as_unitless(orig_nbf.getParticleParameters(p1))
                c2, _, _ = maybe_params_as_unitless(orig_nbf.getParticleParameters(p2))
                if orig_nbf == self._old_nbf:
                    _ = cbf.addBond(hybrid_p1, hybrid_p2,
                                    [c, 0,
                                     s, s,
                                     e, 0,
                                     1, 0,
                                     c1*c2, 0])
                else:
                    _ = cbf.addBond(hybrid_p1, hybrid_p2,
                                    [0, c,
                                     s, s,
                                     0, e,
                                     0, 1,
                                     0, c1*c2])
        return {'dynamic_unique': cbf}

    def _make_mapped_exception_forces(self, **kwargs):
        """make some exception forces that handle the mapped exceptions between the old/new forces.
        For this, I'll query `self._consistent_mapped_params`
        There is a couple of special handlers:
        1. if the old/new term does not change:
            if there is a constraint: place it into a lambda_global-independent, r-independent self-force.
            else (there is not a constraint): place it into a lambda_global-independent force
        2. if the old/new term does change: place into a lambda_global-dependent force;
            here, if one of the terms goes to zero at an endstate, lift it at that endstate
        """
        std_cbf = self._make_standard_cbf(**kwargs)
        alch_cbf = self._make_alch_cbf(**kwargs)
        const_cbf = self._make_standard_constraint_cbf(**kwargs)

        for hybrid_idx_tuple, (old_exc_params, new_exc_params) in self._consistent_mapped_params.items():
            hp1, hp2 = hybrid_idx_tuple
            op1, op2 = self._hybrid_to_old_map[hp1], self._hybrid_to_old_map[hp2]
            np1, np2 = self._hybrid_to_new_map[hp1], self._hybrid_to_new_map[hp2]
            oc1, _, _ = maybe_params_as_unitless(self._old_nbf.getParticleParameters(op1))
            oc2, _, _ = maybe_params_as_unitless(self._old_nbf.getParticleParameters(op2))
            nc1, _, _ = maybe_params_as_unitless(self._new_nbf.getParticleParameters(np1))
            nc2, _, _ = maybe_params_as_unitless(self._new_nbf.getParticleParameters(np2))
            oc_, nc_ = oc1 * oc2, nc1 * nc2
            oc, nc = old_exc_params[0], new_exc_params[0]
            os, ns = old_exc_params[1], new_exc_params[1]
            oe, ne = old_exc_params[2], new_exc_params[2]
            identical = are_nb_params_identical(oc, nc, os, ns, oe, ne, oc_, nc_)
            if identical: # this can go into standard force
                constraint_len = self._constraints_dict.get(hybrid_idx_tuple, -1)
                if constraint_len < 0: # there is no constraint match; place in standard cbf
                    _ = std_cbf.addBond(*hybrid_idx_tuple, [oc, os, oe, oc_])
                else: # there is a constraint; place in constraint force
                    _ = const_cbf.addBond(*hybrid_idx_tuple, [oc, os, oe, oc_, constraint_len])
            else: # these terms are not identical; interpolate between them; lift appropriately
                params = [oc, nc, os, ns, oe, ne, 0, 0, oc_, nc_]
                old_term_zero = np.isclose(oc, 0) and np.isclose(oe, 0) # the old term is zeroed out
                new_term_zero = np.isclose(nc, 0) and np.isclose(ne, 0) # the new term is zeroed out
                if old_term_zero and not new_term_zero: # treat this as unique new
                    params[7] = 1
                elif not old_term_zero and new_term_zero: # treat as unique old
                    params[6] = 1
                else: # either old/new term is zero or neither is zero but params change; there are no singularities
                    pass
                _ = alch_cbf.addBond(*hybrid_idx_tuple, params)

        # now the tricky part is to iterate over the inconsistent parameters and add the _c parameters for consistency
        for orig_nbf, inconsistent_indices in self._inconsistent_mapped_indices.items():
            to_hybrid_map = self._old_to_hybrid_map if orig_nbf == self._old_nbf else self._new_to_hybrid_map
            for inconsistent_idx in inconsistent_indices:
                p1, p2, _, _, _ = orig_nbf.getExceptionParameters(inconsistent_idx)
                c1, _, _ = maybe_params_as_unitless(orig_nbf.getParticleParameters(p1))
                c2, _, _ = maybe_params_as_unitless(orig_nbf.getParticleParameters(p2))
                c_ = c1*c2
                hybrid_p1, hybrid_p2 = to_hybrid_map[p1], to_hybrid_map[p2]
                exc_params = self._exception_parameters[orig_nbf][tuple(sorted([p1, p2]))]
                if orig_nbf == self._old_nbf: # the term is unique old
                    params = [exc_params[0], 0.,
                              exc_params[1], exc_params[1],
                              exc_params[2], 0.,
                              1, 0,
                              c_, 0.]
                else:
                    params = [0., exc_params[0],
                              exc_params[1], exc_params[1],
                              0., exc_params[2],
                              0, 1,
                              0., c_]
                _ = alch_cbf.addBond(hybrid_p1, hybrid_p2, params)

        return {'static': std_cbf, 'dynamic_mapped': alch_cbf, 'static_constraint': const_cbf}

    def make_exceptions(self, **kwargs):
        """call all `openmm.CustomBondForce` generators and return a dict"""
        out_cbf_dict = {}
        for gen in [self._make_inconsistent_exception_force, self._make_unique_exception_force, self._make_mapped_exception_forces]:
            out_cbf_dict.update(gen(**kwargs))
        return out_cbf_dict

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
        krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff ** 3)
        mrf = 4
        nrf = 6
        arfm = (3 * cutoff ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * \
               ((2 * eps_rf + nrf - 1) / (1 + 2 * eps_rf))
        arfn = (3 * cutoff ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * \
               ((2 * eps_rf + mrf - 1) / (1 + 2 * eps_rf))
        crf = ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff) + arfm * \
              cutoff ** mrf + arfn * cutoff ** nrf
        out_dict = {'krf': krf, 'mrf': mrf, 'nrf': nrf,
                    'arfm': arfm, 'arfn': arfn, 'crf': crf,
                    'ONE_4PI_EPS0': self.ONE_4PI_EPS0, 'r_cut': self._cutoff}
        return out_dict

    def make_self_force(self, **unused_kwargs):
        """create the `self` force in a dict to account for particle particle partial charges."""
        bf = openmm.CustomBondForce('')  # space filler for expression
        rf_terms = self._get_rf_terms(**unused_kwargs)
        bf.setUsesPeriodicBoundaryConditions(self._is_periodic)

        aux_template, perBondParameters, global_parameters = self._get_aux_self_terms(**unused_kwargs)
        for bond_param in perBondParameters:  # add per particle params
            bf.addPerBondParameter(bond_param)
        all_global_params = copy.deepcopy(self.NB_GLOBAL_PARAMETERS)
        all_global_params.update(global_parameters)
        for gp_name, gp_val in all_global_params.items():  # add global params
            bf.addGlobalParameter(gp_name, gp_val)
        energy_fn = ' '.join(aux_template).format(**rf_terms).format(**self.NB_SOFTCORE_GLOBAL_PARAMETERS)
        bf.setEnergyFunction(energy_fn)

        # now add the terms
        for i in range(self._num_hybrid_particles):
            try:
                old_c1, _, _ = self._old_nbf.getParticleParameters(
                    self._hybrid_to_old_map[i])
            except:  # it's unique new
                old_c1 = 0.
            try:
                new_c1, _, _ = self._new_nbf.getParticleParameters(
                    self._hybrid_to_new_map[i])
            except:  # it's unique old
                new_c1 = 0.
            _ = bf.addBond(i, i, [old_c1 * old_c1, new_c1 * new_c1])
        return {'self': bf}
