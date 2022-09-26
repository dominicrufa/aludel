"""utility functions for generating hybrid systems; let's rename this `.py`;
it makes me sad"""

import openmm
from openmm import app, unit
import numpy as np
import copy
from typing import Any, Tuple, Dict, Iterable, Callable

# define basic alchemical transfer expression

V1_ATM_EXPR_TEMPLATE = [
'U0 + (({lambda2} - {lambda1})/{alpha}) * log_term + {lambda2}*u_sc + {w0};',
'log_term = c + log(exp(-c) + exp(exponand-c));',
'c = max(0, exponand);',
'exponand = -{alpha}*(u_sc - {u0});',
'u_sc = select(soften_bool, u_soft, u);',
'soften_bool = step(u - {u_cut});',
'u_soft = ({u_max} - {u_cut}) * f_sc + {u_cut};',
'f_sc = (z_y^{a} - 1)/(z_y^{a} + 1);',
'z_y = 1 + 2*y_by_a + 2*(y_by_a^2);',
'y_by_a = y / {a};',
'y = (u - {u_cut}) / ({u_max} - {u_cut});',
'u = U1 - U0;',
'U1 = select(past_half, U0_static, U1_static);',
'U0 = select(past_half, U1_static, U0_static);',
'past_half = step({time} - 0.5);'
]

# # overwrite
# V1_ATM_EXPR_TEMPLATE = [
#     'U1_static;',
# ]

V1_ATM_EXPR = ''.join(V1_ATM_EXPR_TEMPLATE)

ATM_COLLECTIVE_VARS = ['U0_static', 'U1_static']

V1_ATM_DEFAULT_GLOBAL_PARAMS = {
'time': 0.,
'lambda1': 0.,
'lambda2': 0.,
'alpha': 0.1,
'u_cut': 200., # check this again in comparison to `_u0`
'u0': 100.,
'u_max': 400.,
'a': 0.0625,
'w0': 0.,
}


# define per particle and global parameters for valence forces
VALENCE_FORCE_STATIC_EXPR = {
'HarmonicBondForce': {
    'standard_add_term_expr': 'addBond',
    'query_num_terms_expr': 'getNumBonds',
    'query_params_expr': 'getBondParameters',
    'standard_force_obj': openmm.HarmonicBondForce,
    'get_param_particle_indices': 2,
    'parameter_setter_fn': 'setBondParameters',
    'num_parameters': 2,
    },
'HarmonicAngleForce': {
    'standard_add_term_expr': 'addAngle',
    'query_num_terms_expr': 'getNumAngles',
    'query_params_expr': 'getAngleParameters',
    'standard_force_obj': openmm.HarmonicAngleForce,
    'get_param_particle_indices': 3,
    'parameter_setter_fn': 'setAngleParameters',
    'num_parameters': 2,
    },
'PeriodicTorsionForce': {
    'standard_add_term_expr': 'addTorsion',
    'query_num_terms_expr': 'getNumTorsions',
    'query_params_expr': 'getTorsionParameters',
    'standard_force_obj': openmm.PeriodicTorsionForce,
    'get_param_particle_indices': 4,
    'parameter_setter_fn': 'setTorsionParameters',
    'num_parameters': 3,
    }
}


# utilities

def translate_standard_valence_force(
    old_force: openmm.Force,
    new_force: openmm.Force,
    old_to_hybrid_map: Iterable[int],
    new_to_hybrid_map: Iterable[int],
    unique_old_atoms: Iterable[int],
    unique_new_atoms: Iterable[int],
    **kwargs) -> Tuple[openmm.Force, openmm.Force, Dict[str, Iterable[int]]]:
    """translate a standard valence force; will return:
        1. the translated old/new forces (U0_static/U1_static)
        2. a dict of keys ['U0_static', 'U1_static']
            containing the (artificial) indices of terms that were added to
            the output valence forces to retain valence forces on unique
            old/new terms. This is primarily used for bookkeeping purposes

    we will also bookkeep
    1. a running tally of the term indices for each out force for
        `artificial_terms`; i.e., terms that are added as static because they
        exist containing a `unique` particle in the opposite force.
    """
    old_forcename = old_force.__class__.__name__
    new_forcename = new_force.__class__.__name__
    unique_term_indices = {'old': [], 'new': []} # bookkeep unique terms
    core_terms = {}

    force_util_dict = VALENCE_FORCE_STATIC_EXPR[old_forcename] # get utility dict
    num_particles_per_term = force_util_dict['get_param_particle_indices']
    assert old_forcename == new_forcename, f"""
    old force {old_forcename} doesn't match new force {new_forcename}
    """
    U0_force, U1_force = copy.deepcopy(old_force), copy.deepcopy(new_force)

    # now we just need to convert these to hybrid indices and document the unique terms
    for out_force, force_label in zip([U0_force, U1_force], ['old', 'new']):
        num_terms = getattr(out_force,
                            force_util_dict['query_num_terms_expr'])()
        query_params_fn = getattr(out_force,
                                force_util_dict['query_params_expr'])
        uniques_to_query = unique_old_atoms if force_label == 'old' else unique_new_atoms
        to_hybrid_map = old_to_hybrid_map if force_label == 'old' else new_to_hybrid_map
        parameter_setter_fn = getattr(out_force,
                                      force_util_dict['parameter_setter_fn'])
        for term_idx in range(num_terms):
            all_parameters = query_params_fn(term_idx)
            particles = all_parameters[:num_particles_per_term]
            per_term_parameters = all_parameters[num_particles_per_term:]
            if any([_idx in uniques_to_query for _idx in particles]):
                unique_term_indices[force_label].append(term_idx)
            hybrid_indices = [to_hybrid_map[_idx] for _idx in particles]
            parameter_setter_fn(term_idx, *hybrid_indices, *per_term_parameters)

    # finally, query through the unique term indices and add the term to the
    # opposite force; we want to track these so we can energy bookkeep afterward.
    artifical_terms = {'U0_static': [], 'U1_static': []}
    for unique_label, term_indices in unique_term_indices.items():
        out_force = new_force if unique_label=='old' else old_force
        query_force = old_force if unique_label=='old' else new_force
        write_force = U1_force if unique_label=='old' else U0_force
        query_params_fn = getattr(query_force,
                        force_util_dict['query_params_expr'])
        artificial_term_iterable = artifical_terms['U1_static'] if unique_label=='old' \
            else artifical_terms['U0_static']
        to_hybrid_map = old_to_hybrid_map if unique_label == 'old' else new_to_hybrid_map
        term_add_fn = getattr(write_force,
                              force_util_dict['standard_add_term_expr'])
        for term_idx in term_indices:
            all_parameters = query_params_fn(term_idx)
            particles = all_parameters[:num_particles_per_term]
            per_term_parameters = all_parameters[num_particles_per_term:]
            hybrid_indices = [to_hybrid_map[_idx] for _idx in particles]
            artificial_term = term_add_fn(*hybrid_indices, *per_term_parameters)
            artificial_term_iterable.append(artificial_term)

    return U0_force, U1_force, artifical_terms

def sort_indices_to_str(indices: Iterable[int]) -> str:
    sorted_indices = sorted(indices)
    return '.'.join([str(_q) for _q in sorted_indices])

def params_as_unitless(parameters: Iterable) -> Iterable:
    _out = [_val.value_in_unit_system(unit.md_unit_system) \
               for _val in parameters]
    return _out

def make_exception_param_from_particles(_force_to_query, particles):
    p1_particle_params = params_as_unitless(_force_to_query.getParticleParameters(particles[0]))
    p2_particle_params = params_as_unitless(_force_to_query.getParticleParameters(particles[1]))
    cp = p1_particle_params[0] * p2_particle_params[0]
    s = (p1_particle_params[1] + p2_particle_params[1])/2.
    e = np.sqrt(p1_particle_params[2]*p2_particle_params[2])
    return cp, s, e


def translate_standard_nonbonded_force(
    old_nbf: openmm.NonbondedForce,
    new_nbf: openmm.NonbondedForce,
    num_hybrid_particles: int,
    old_to_hybrid_map: Dict[int, int],
    new_to_hybrid_map: Dict[int, int],
    unique_old_atoms: Iterable[int],
    unique_new_atoms: Iterable[int],
    unique_old_exc_offset_gp: str,
    unique_new_exc_offset_gp: str,
    exc_offset_gp: str,
    particle_offset_gp: str,
    **kwargs) -> Tuple[openmm.NonbondedForce, openmm.NonbondedForce]:
    """
    translate an old/new standard nonbonded force into a PME-treated
    `U0_static`, `U1_static` for ATM RBFEs.
    NOTE: this method effectively translates `old_nbf` and `new_nbf`
        to hybrid indices whilst retaining nonbonded exceptions
        associated with `unique` new and old atoms since these are
        considered "valence" terms that should be immutable.
    Arguments:
        old_nbf: old `openmm.NonbondedForce`
        new_nbf: new `openmm.NonbondedForce`
        num_hybrid_particles: number of particles in the hybrid system
        old_to_hybrid_map: dictionary mapping old particles to
            hybrid indices
        new_to_hybrid_map: dictionary of mapping new particles to
            hybrid indices
        unique_old_atoms: old particle indices that only exist
            in old system
        unique_new_atoms: new particle indices that only exist
            in new system
        unique_old_exc_offset_gp: global param name that turns on (1.)
            or off (0.) nonbonded exceptions associated with the unique
            old atoms
        unique_new_exc_offset_gp: global param name that turns on (1.)
            or off (0.) nonbonded exceptions associated with the unique
            new atoms
        exc_offset_gp: global param that translates mapped exceptions
            from `old_nbf` to `new_nbf` from 0. to 1., respectively
        particle_offset_gp: global param that translates mapped particle
            indices from `old_nbf` to `new_nbf` from 0. to 1.,
            respectively
    Returns
        U0_static: `openmm.NonbondedForce` representing `old_nbf`
            translated to hybrid indices and containing exceptions
            associated with `unique_new_atoms`
        U1_static: `openmm.NonbondedForce` representing `new_nbf`
            translated to hybrid_indices and containing exceptions
            associated with `unique_old_atoms`
    """

    # make U_static and add global parameters
    U_static = copy.deepcopy(old_nbf)
    for gp in [unique_old_exc_offset_gp, unique_new_exc_offset_gp]:
        U_static.addGlobalParameter(gp, 1.)
    for gp in [exc_offset_gp, particle_offset_gp]:
        U_static.addGlobalParameter(gp, 0.)
    num_old_particles = old_nbf.getNumParticles()
    num_new_particles = new_nbf.getNumParticles()
    assert all([key==val for key, val in old_to_hybrid_map.items()]), \
        f"""nbf translation requirement for consistency"""
    assert num_old_particles == len(old_to_hybrid_map)

    hybrid_to_old_map = {val:key for key, val in old_to_hybrid_map.items()}
    hybrid_to_new_map = {val:key for key, val in new_to_hybrid_map.items()}

    # add extra particles
    particle_difference = num_hybrid_particles - num_old_particles
    [U_static.addParticle(0.,0.,0.) for _ in range(particle_difference)]

    # first thing to do is gather all of the nonbonded exceptions
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

    # now, iterate through the exceptions
    for _force in [old_nbf, new_nbf]:
        num_exceptions = _force.getNumExceptions()
        opp_force = old_nbf if _force == new_nbf else new_nbf
        to_hybrid_map = old_to_hybrid_map if _force == old_nbf \
            else new_to_hybrid_map
        to_opposite_orig_map = hybrid_to_new_map if _force == old_nbf \
            else hybrid_to_old_map
        opp_exc_dict_to_query = exception_data[new_nbf] if \
            _force == old_nbf else exception_data[old_nbf]
        uniques = unique_old_atoms if _force == old_nbf \
            else unique_new_atoms
        unique_param_offset_gp = unique_old_exc_offset_gp \
            if _force == old_nbf else unique_new_exc_offset_gp
        for orig_exc_idx in range(num_exceptions):
            orig_exc_params = _force.getExceptionParameters(orig_exc_idx)
            orig_indices = orig_exc_params[:2]
            orig_nonidx_params = params_as_unitless(orig_exc_params[2:])
            hybrid_inds = [to_hybrid_map[_q] for _q in orig_indices]
            sorted_hybrid_inds_str = sort_indices_to_str(hybrid_inds)
            contains_unique = len(
                    set(orig_indices).intersection(set(uniques))) > 0
            try: # get the exception from the opposite system
                opp_exc_idx = opp_exc_dict_to_query[sorted_hybrid_inds_str]
            except Exception as e: # the exception idx doesn't exist;
                opp_exc_idx = -1
            if opp_exc_idx == -1:
                # that means the particles included are unique or the
                # exception is simply not in the opposing system
                if contains_unique:
                    # zero the params and make an offset for unique news
                    new_exc_idx = U_static.addException(*hybrid_inds,
                        0., 0., 0., replace=True)
                    new_exc_offset_idx = U_static.addExceptionParameterOffset(
                        unique_param_offset_gp, new_exc_idx, *orig_nonidx_params)
                else:
                    raise Exception(f"""this is not well-tested;
                        reconsider your mapping, please.""")
                    # the opposite params must be queried from particle params
                    opp_particle_indices = [to_opposite_orig_map[_q] for
                                            _q in hybrid_inds]
                    cp, s, e = make_exception_param_from_particles(opp_force,
                                                                   opp_particle_indices)
                    _scales = [_new - _old for _old, _new in zip(orig_nonidx_params,
                                                                [cp, s, e])]
                    new_exc_offset_idx = U_static.addExceptionParameterOffset(
                        exc_offet_gp, orig_exc_idx, _scales)
            else: # this means that the exception _is_
                # mapped in the opposite system
                # only write the offset for the first iter in the for (old_nbf)
                # and if the the parameters are not the same (otherwise it would
                # be redundant.)
                opposite_parameters = params_as_unitless(
                        opp_force.getExceptionParameters(opp_exc_idx)[2:])
                if _force == old_nbf and not np.allclose(
                    orig_nonidx_params, opposite_parameters):
                    opposite_parameters = params_as_unitless(
                        opp_force.getExceptionParameters(opp_exc_idx)[2:])
                    _scales = [_new - _old for _old, _new in zip(orig_nonidx_params,
                                                                opposite_parameters)]
                    new_exc_offset_idx = U_static.addExceptionParameterOffset(
                        exc_offset_gp, orig_exc_idx, *_scales)

    # then add exceptions between all unique new/old
    for unique_new_idx in unique_new_atoms:
        unique_new_hybr_idx = new_to_hybrid_map[unique_new_idx]
        for unique_old_idx in unique_old_atoms:
            unique_old_hybrid_idx = old_to_hybrid_map[unique_old_idx]
            _ = U_static.addException(unique_old_hybrid_idx,
                unique_new_hybr_idx, 0., 0., 0.)

    # exceptions are handled
    # iterate over particles...
    for old_idx, hybrid_idx in old_to_hybrid_map.items(): # redundant
        # this is redundant because of first assert statement
        old_params = old_nbf.getParticleParameters(old_idx)
        U_static.setParticleParameters(hybrid_idx, *old_params)
        if old_idx in unique_old_atoms: # create an offset
            _scales = [-1*_i for _i in params_as_unitless(old_params)]
            _ = U_static.addParticleParameterOffset(
                particle_offset_gp, hybrid_idx, *_scales)
        try:
            new_particle_idx = hybrid_to_new_map[hybrid_idx]
        except:
            new_particle_idx = -1
        if new_particle_idx >=0: # it is mapped; make offset
            new_params = new_nbf.getParticleParameters(new_particle_idx)
            _scales = [_new - _old for _old, _new in
                zip(params_as_unitless(old_params), params_as_unitless(new_params))]
            _ = U_static.addParticleParameterOffset(
                particle_offset_gp, hybrid_idx, *_scales)
        else:
            assert old_idx in unique_old_atoms

    for new_idx in unique_new_atoms:
        new_params = new_nbf.getParticleParameters(new_idx)
        hybrid_idx = new_to_hybrid_map[new_idx]
        _ = U_static.addParticleParameterOffset(
            particle_offset_gp, hybrid_idx, *new_params)

    # now pull it all together
    U0_static = copy.deepcopy(U_static)
    U1_static = copy.deepcopy(U_static)

    for gp_idx in range(U0_static.getNumGlobalParameters()):
        gp_name = U0_static.getGlobalParameterName(gp_idx)
        gp_val = U0_static.getGlobalParameterDefaultValue(gp_idx)
        if np.isclose(gp_val, 0.): # then U1_static gets turned to 1
            U1_static.setGlobalParameterDefaultValue(gp_idx, 1.)
        U0_static.setGlobalParameterName(gp_idx, f"U0_static_" + gp_name)
        U1_static.setGlobalParameterName(gp_idx, f"U1_static_" + gp_name)
    return U0_static, U1_static

def get_hybrid_positions(old_positions: unit.Quantity,
                         new_positions: unit.Quantity,
                         num_hybrid_particles: int,
                         old_to_hybrid_map: Dict[int, int],
                         new_to_hybrid_map: Dict[int, int],
                         mapped_positions_on_old: bool=False,
                         **unused_kwargs) -> unit.Quantity:
    """get hybrid positions from old/new positions; `openmm`-amenable;
    `mapped_positions_on_old` will write mapped positions to the old particle indices;
    otherwise, will map to new."""
    hybrid_positions = np.zeros((num_hybrid_particles, 3))
    old_pos_sans_units = old_positions.value_in_unit_system(unit.md_unit_system)
    new_pos_sans_units = new_positions.value_in_unit_system(unit.md_unit_system)
    to_hybrid_maps = [old_to_hybrid_map, new_to_hybrid_map]
    from_position_cache = [old_pos_sans_units, new_pos_sans_units]
    if mapped_positions_on_old:
        to_hybrid_maps = to_hybrid_maps[::-1]
        from_position_cache = from_position_cache[::-1]

    for to_hybrid_map, from_positions in zip(to_hybrid_maps, from_position_cache):
        for orig_idx, hybrid_idx in to_hybrid_map.items():
            hybrid_positions[hybrid_idx,:] = from_positions[orig_idx,:]
    return hybrid_positions * unit.nanometers

def make_CustomCVForce(
    global_parameters: Dict[str, float],
    energy_fn_expression: str,
    energy_fn_collective_vars: Dict[float, openmm.Force],
    **unused_kwargs):
    cv = openmm.CustomCVForce(energy_fn_expression)
    for energy_fn, force in energy_fn_collective_vars.items():
        cv.addCollectiveVariable(energy_fn, copy.deepcopy(force))
    for param_name, param_value in global_parameters.items():
        cv.addGlobalParameter(param_name, param_value)
    return cv

class BaseHybridSystemFactory(object):
    """
    base class for generating a hybrid system object
    """
    _allowed_force_names = ['HarmonicBondForce',
                            'HarmonicAngleForce',
                            'PeriodicTorsionForce',
                            'NonbondedForce',
                            'MonteCarloBarostat'
                            ]
    def __init__(
        self: Any,
        old_system: openmm.System, # old_system
        new_system: openmm.System, # new system
        old_to_new_atom_map: Dict[int, int],
        unique_new_atoms: Iterable[int],
        unique_old_atoms: Iterable[int],
        atm_expression_template: str=V1_ATM_EXPR,
        atm_default_global_parameters: Dict[str, float]=V1_ATM_DEFAULT_GLOBAL_PARAMS,
        nbf_unique_old_exc_offset_gp: str='nbf_unique_old_exc_offset_gp',
        nbf_unique_new_exc_offset_gp: str='nbf_unique_new_exc_offset_gp',
        nbf_exc_offset_gp: str='nbf_exc_offset_gp',
        nbf_particle_offset_gp: str='nbf_particle_offset_gp',
        **kwargs):

        self._old_system = copy.deepcopy(old_system)
        self._new_system = copy.deepcopy(new_system)

        self._old_to_new_atom_map = copy.deepcopy(old_to_new_atom_map)

        self._unique_old_atoms = copy.deepcopy(unique_old_atoms)
        self._unique_new_atoms = copy.deepcopy(unique_new_atoms)

        self._hybrid_system = openmm.System()

        self._atm_expression_template = atm_expression_template
        self._atm_collective_variable_names = ATM_COLLECTIVE_VARS
        self._atm_default_global_parameters = atm_default_global_parameters

        self._nbf_unique_old_exc_offset_gp = nbf_unique_old_exc_offset_gp
        self._nbf_unique_new_exc_offset_gp = nbf_unique_new_exc_offset_gp
        self._nbf_exc_offset_gp = nbf_exc_offset_gp
        self._nbf_particle_offset_gp = nbf_particle_offset_gp

        # now call setup fns
        self._add_particles_to_hybrid(**kwargs)
        self._get_force_dicts(**kwargs) # render force dicts for new/old sys
        self._assert_allowable_forces(**kwargs)
        self._copy_box_vectors(**kwargs) # copy box vectors
        self._handle_constraints(**kwargs)
        self._handle_virtual_sites(**kwargs)
        self._equip_valence_forces(**kwargs)
        self._equip_nonbonded_force(**kwargs)

        if self._hybrid_system.usesPeriodicBoundaryConditions():
            self._copy_barostat(**kwargs) # copy barostat


    def _get_force_dicts(self, **unused_kwargs):
        """make a dict of force for each system and set attrs"""
        for key, sys in zip(['_old_forces', '_new_forces'],
                            [self._old_system, self._new_system]):
            setattr(self,
                key,
                {force.__class__.__name__: force for force in sys.getForces()}
                )

    def _assert_allowable_forces(self, **unused_kwargs):
        """this method ensures that the two systems are effectively identical
        at the `System` level; importantly, we assert that:
        1. the number/name of forces in each system is identical
        2. the force name in each system is in the 'allowed_forcenames'
        """
        # make sure that each force in each system is allowed
        for force_dict in [self._old_forces, self._new_forces]:
            try:
                for force_name in force_dict.keys():
                    assert force_name in self._allowed_force_names
            except Exception as e:
                raise NameError(f"""
                    In querying force dict, assertion failed with {e}
                    """)

    def _copy_barostat(self, **unused_kwargs):
        if "MonteCarloBarostat" in self._old_forces.keys():
            barostat = copy.deepcopy(self._old_forces["MonteCarloBarostat"])
            self._hybrid_system.addForce(barostat)

    def _copy_box_vectors(self, **unused_kwargs):
        box_vectors = self._old_system.getDefaultPeriodicBoxVectors()
        self._hybrid_system.setDefaultPeriodicBoxVectors(*box_vectors)

    def _handle_constraints(self, **unused_kwargs):
        """copy constraints; check to make sure it doesn't change"""
        constraint_lengths = {}
        for system_name in ['old', 'new']:
            sys = getattr(self, f"_{system_name}_system")
            hybrid_map = getattr(self, f"_{system_name}_to_hybrid_map")
            for constraint_idx in range(sys.getNumConstraints()):
                a1, a2, length = sys.getConstraintParameters(constraint_idx)
                hybr_atoms = tuple(sorted([hybrid_map[a1], hybrid_map[a2]]))
                if hybr_atoms not in constraint_lengths.keys():
                    self._hybrid_system.addConstraint(*hybr_atoms, length)
                    constraint_lengths[hybr_atoms] = length
                else:
                    if constraint_lengths[hybr_atoms] != length:
                        raise Exception(f"""
                        Constraint length is changing for atoms {hybr_atoms}
                        in hybrid system: old is
                        {constraint_lengths[hybr_atoms]} and new is
                        {length}
                        """)
    def _handle_virtual_sites(self, **unused_kwargs):
        """for now, assert that none of the particle are virtual sites"""
        for system_name in ['old', 'new']:
            sys = getattr(self, f"_{system_name}_system")
            num_ps = sys.getNumParticles()
            for idx in range(num_ps):
                if sys.isVirtualSite(idx):
                    raise Exception(f"""
                    system {system_name} particle {idx} is a virtual site
                    but virtual sites are not currently supported.
                    """)

    def _add_particles_to_hybrid(self, **unused_kwargs):
        """copy all old system particles to hybrid with identity mapping"""
        self._old_to_hybrid_map = {}
        self._new_to_hybrid_map = {}

        old_num_particles = self._old_system.getNumParticles()
        new_num_particles = self._new_system.getNumParticles()
        for idx in range(old_num_particles): # iterate over old particles
            mass_old = self._old_system.getParticleMass(idx)
            if idx in self._old_to_new_atom_map.keys():
                idx_in_new_sys = self._old_to_new_atom_map[idx]
                mass_new = self._new_system.getParticleMass(idx_in_new_sys)
                mix_mass = (mass_old + mass_new) / 2
            else:
                mix_mass = mass_old
            hybrid_idx = self._hybrid_system.addParticle(mix_mass)
            self._old_to_hybrid_map[idx] = hybrid_idx

        # make assertion on old particles map equivalence
        assert {key==val for key, val in self._old_to_hybrid_map.items()}, f"""
        old particle to hybrid particle map is not equivalent"""

        # make assertion on old particles
        assert len(self._old_to_hybrid_map) == old_num_particles, f"""
        there is a convention mistake; the `self._old_to_hybrid_map`
        has {len(self._old_to_hybrid_map)} particles, but the old
        system has {old_num_particles} particles"""

        # make the `hybrid_to_old_map`
        self._hybrid_to_old_map = {value: key for key, value in
                                self._old_to_hybrid_map.items()}

        # make `self._hybrid_to_new_map`;
        # first, since all of the old indices are equiv to hybrid indices...
        self._hybrid_to_new_map = copy.deepcopy(self._old_to_new_atom_map)

        for idx in self._unique_new_atoms: # iterate over new particles
            mass = self._new_system.getParticleMass(idx)
            hybrid_idx = self._hybrid_system.addParticle(mass)
            self._hybrid_to_new_map[hybrid_idx] = idx

        # make assertion on new particles
        new_num_particles = self._new_system.getNumParticles()
        assert len(self._hybrid_to_new_map) == new_num_particles, f"""
        there is a convention mistake; the `self._new_to_hybrid_map`
        has {len(self._hybrid_to_new_map)} particles, but the new
        system has {new_num_particles} particles"""

        # make the `new_to_hybrid_map`
        self._new_to_hybrid_map = {value: key for key, value in
                                  self._hybrid_to_new_map.items()}



    def _make_valence_forces(self, **kwargs):
        """construct static/perturbation valence forces and return a
        dictionary of such force for each force name by calling
        `translate_standard_valence_force` on each valence force object
        """
        allowed_valence_forcenames = list(VALENCE_FORCE_STATIC_EXPR.keys())
        all_forcenames = list(self._old_forces.keys()) + \
            list(self._new_forces.keys())
        all_forcenames_set = set(all_forcenames)
        out_force_dict = {}

        # iterate over the set of all forces
        for force_name in all_forcenames_set:
            if force_name in allowed_valence_forcenames:
                old_force = self._old_forces[force_name]
                new_force = self._new_forces[force_name]
                U0_f, U1_f, artifical_terms = translate_standard_valence_force(
                    old_force=old_force,
                    new_force=new_force,
                    old_to_hybrid_map=self._old_to_hybrid_map,
                    new_to_hybrid_map=self._new_to_hybrid_map,
                    unique_old_atoms=self._unique_old_atoms,
                    unique_new_atoms=self._unique_new_atoms,
                    **kwargs)
                out_force_dict[force_name] = {
                    'U0_static': U0_f,
                    'U1_static': U1_f,
                    'artificial_term_dict': artifical_terms}
        return out_force_dict

    def _equip_valence_forces(self, **kwargs):
        """make `openmm.CustomCVForce` for valence forces and equip
        to `self._hybrid_system`"""
        atm_global_param_names = {key: key for key in
                                  self._atm_default_global_parameters.keys()}
        energy_expr = self._atm_expression_template.format(
            **atm_global_param_names) # register params in place

        # valence forces
        valence_force_dict = self._make_valence_forces(**kwargs)
        self._artificial_valence_force_terms = {} # for bookkeeping
        for valence_force_name, valence_force_terms in valence_force_dict.items():
            # equip the static valence force terms for bookkeeping
            self._artificial_valence_force_terms[valence_force_name] = \
                valence_force_terms['artificial_term_dict']

            # then construct and equip CustomCVForce
            coll_vars = {_key: _val for _key, _val in valence_force_terms.items()
                        if _key in self._atm_collective_variable_names}
            cv = make_CustomCVForce(
                global_parameters = self._atm_default_global_parameters,
                energy_fn_expression = energy_expr,
                energy_fn_collective_vars = coll_vars,
                **kwargs)
            self._hybrid_system.addForce(cv)

    def _equip_nonbonded_force(self,
                               nonbonded_force_name: str='NonbondedForce',
                               **kwargs):
        """make `openmm.CustomCVForce` for nonbonded force and equip
        to `self._hybrid_system`"""
        atm_global_param_names = {key: key for key in
                          self._atm_default_global_parameters.keys()}
        energy_expr = self._atm_expression_template.format(
            **atm_global_param_names) # register params in place

        # NonbondedForce
        old_nbf = self._old_forces[nonbonded_force_name]
        new_nbf = self._new_forces[nonbonded_force_name]
        U0_nbf, U1_nbf = translate_standard_nonbonded_force(
            old_nbf = old_nbf,
            new_nbf = new_nbf,
            num_hybrid_particles = self._hybrid_system.getNumParticles(),
            old_to_hybrid_map = self._old_to_hybrid_map,
            new_to_hybrid_map = self._new_to_hybrid_map,
            unique_old_atoms = self._unique_old_atoms,
            unique_new_atoms = self._unique_new_atoms,
            unique_old_exc_offset_gp = self._nbf_unique_old_exc_offset_gp,
            unique_new_exc_offset_gp = self._nbf_unique_new_exc_offset_gp,
            exc_offset_gp = self._nbf_exc_offset_gp,
            particle_offset_gp = self._nbf_particle_offset_gp,
            **kwargs)
        # then construct and equip CustomCVForce
        coll_vars = {'U0_static': U0_nbf, 'U1_static': U1_nbf}
        nbf_cv = make_CustomCVForce(
            global_parameters = self._atm_default_global_parameters,
            energy_fn_expression = energy_expr,
            energy_fn_collective_vars = coll_vars,
            **kwargs)
        self._hybrid_system.addForce(nbf_cv)

    @property
    def hybrid_system(self):
        return copy.deepcopy(self._hybrid_system)

    @property
    def old_system(self):
        return copy.deepcopy(self._old_system)

    @property
    def new_system(self):
        return copy.deepcopy(self._new_system)
