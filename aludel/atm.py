import openmm
from openmm import app, unit
import numpy as np
import copy
from typing import Any, Tuple, Dict, Iterable, Callable
from aludel.utils import maybe_params_as_unitless, sort_indices_to_str

# utilities

def make_exception_param_from_particles(_force_to_query, particles):
    p1_particle_params = maybe_params_as_unitless(_force_to_query.getParticleParameters(particles[0]))
    p2_particle_params = maybe_params_as_unitless(_force_to_query.getParticleParameters(particles[1]))
    cp = p1_particle_params[0] * p2_particle_params[0]
    s = (p1_particle_params[1] + p2_particle_params[1])/2.
    e = np.sqrt(p1_particle_params[2]*p2_particle_params[2])
    return cp, s, e

def translate_standard_valence_force_to_custom(
    old_force: openmm.Force, # old valence force
    new_force: openmm.Force, # new valence force
    old_to_hybrid_map: Iterable[int],
    new_to_hybrid_map: Iterable[int],
    unique_old_atoms: Iterable[int],
    unique_new_atoms: Iterable[int],
    force_utils: Dict[str, Any],
    custom_expression: str,
    global_parameters: Dict[str, float],
    static_del: Iterable[int] = [1, 0, 0, 0, 0],
    old_del: Iterable[int] = [0, 1, 0, 0, 0],
    new_del: Iterable[int] = [0, 0, 1, 0, 0],
    unique_old_del: Iterable[int] = [0, 0, 0, 1, 0],
    unique_new_del: Iterable[int] = [0, 0, 0, 0, 1],
    **kwargs) -> openmm.Force:
    """
    translate a standard valence force"""
    num_particles = force_utils['num_particles']

    # make the custom valence force and add per term params
    custom_expression = custom_expression + force_utils['U_valence']
    U_out = force_utils['custom_force'](custom_expression)
    is_periodic = old_force.usesPeriodicBoundaryConditions()
    _ = U_out.setUsesPeriodicBoundaryConditions(is_periodic)
    for gp_name, gp_val in global_parameters.items(): # add per particle params
        U_out.addGlobalParameter(gp_name, gp_val)

    add_per_term_param_fn = getattr(U_out, force_utils['add_per_term_param'])
    for _term in force_utils['per_term_params']:
        _ = add_per_term_param_fn(_term)

    # get the setTerm/addTerm/query methods
    term_adder_fn = getattr(U_out, force_utils['addTerm'])
    term_setter_fn = getattr(U_out, force_utils['setTerm'])
    query_U_out_by_idx_fn = getattr(U_out, force_utils['query_params'])

    hybr_idx_to_term_dict = {} # record dict of added terms
    for force in [old_force, new_force]:
        query_term_by_idx_fn = getattr(force, force_utils['query_params'])
        num_terms = getattr(force, force_utils['query_num_terms'])()
        uniques = unique_old_atoms if force == old_force else unique_new_atoms
        to_hybrid_map = old_to_hybrid_map if force == old_force else new_to_hybrid_map
        for idx in range(num_terms):
            all_params = query_term_by_idx_fn(idx) # full param spec
            orig_indices = all_params[:num_particles] # get the original indices
            hybr_indices = [to_hybrid_map[_q] for _q in orig_indices] # map to hybrid indices
            unitless_term_params = maybe_params_as_unitless(all_params[num_particles:]) # make the terms unitless
            hybr_ind_str = '.'.join([str(_q) for _q in hybr_indices]) # make hybrid indices strings
            # check if contains uniques first
            if len(set(orig_indices).intersection(uniques)) > 0:
                # unique new/old particles included
                _del = unique_old_del if force == old_force else unique_new_del
                _term_idx = term_adder_fn(*hybr_indices,
                unitless_term_params + _del)
            else: # no unique particles in this.
                if force == old_force: # iterate over this first,
                    # make it old; adjust this afterward if it is
                    # encountered in `force == new_force`
                    _term_idx = term_adder_fn(*hybr_indices, unitless_term_params + old_del)
                    try: # try to append it to the recorder dict
                        hybr_idx_to_term_dict[hybr_ind_str].append(_term_idx)
                    except Exception as e: # if it is not in the keys, make a new one
                        hybr_idx_to_term_dict[hybr_ind_str] = [_term_idx]
                else: # force == new_force
                    rev_hybr_ind_str = '.'.join([str(_q) for _q in hybr_indices[::-1]])
                    try: # try to query the recorder dict
                        try: # query as the str exists
                            match_term_indices = hybr_idx_to_term_dict[hybr_ind_str]
                        except: # query backward str
                            match_term_indices = hybr_idx_to_term_dict[rev_hybr_ind_str]
                    except Exception as e: # there are no match terms
                        match_term_indices = []
                    if len(match_term_indices) == 0:
                        # this is a mapped term that is new
                        _term_idx = term_adder_fn(*hybr_indices,
                            unitless_term_params + new_del)
                    else: # there is at least 1 idx match; now match parameters
                        param_match = False
                        for match_term_idx in match_term_indices: # iterate over matches
                            match_params = query_U_out_by_idx_fn(match_term_idx)
                            non_delimiter_match_params = match_params[-1][:-5]
                            if np.allclose(list(unitless_term_params), list(non_delimiter_match_params)):
                                # these terms match, and we can make them static...
                                param_match = True
                                break
                        if param_match: # there is a term with _exact_ parameters
                            term_setter_fn(match_term_idx, *hybr_indices, list(non_delimiter_match_params) + static_del)
                        else: # there is no term with _exact_ parametes; add to new
                            _ = term_adder_fn(*hybr_indices, unitless_term_params + new_del)
    return U_out

def translate_standard_nonbonded_force(
    old_nbf: openmm.NonbondedForce,
    new_nbf: openmm.NonbondedForce,
    num_hybrid_particles: int,
    old_to_hybrid_map: Dict[int, int],
    new_to_hybrid_map: Dict[int, int],
    unique_old_atoms: Iterable[int],
    unique_new_atoms: Iterable[int],
    unique_old_exception_switch: str,
    unique_new_exception_switch: str,
    exception_offset: str,
    particle_offset: str,
    **kwargs) -> Tuple[openmm.NonbondedForce, openmm.NonbondedForce]:

    # make U_static and add global parameters
    U_static = copy.deepcopy(old_nbf)
    for gp in [unique_old_exception_switch, unique_new_exception_switch]:
        U_static.addGlobalParameter(gp, 1.)
    for gp in [exception_offset, particle_offset]:
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
        unique_param_offset_gp = unique_old_exception_switch \
            if _force == old_nbf else unique_new_exception_switch
        for orig_exc_idx in range(num_exceptions):
            orig_exc_params = _force.getExceptionParameters(orig_exc_idx)
            orig_indices = orig_exc_params[:2]
            orig_nonidx_params = maybe_params_as_unitless(orig_exc_params[2:])
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
                        exception_offset, orig_exc_idx, _scales)
            else: # this means that the exception _is_
                # mapped in the opposite system
                # only write the offset for the first iter in the for (old_nbf)
                # and if the the parameters are not the same (otherwise it would
                # be redundant.)
                opposite_parameters = maybe_params_as_unitless(
                        opp_force.getExceptionParameters(opp_exc_idx)[2:])
                if _force == old_nbf and not np.allclose(
                    orig_nonidx_params, opposite_parameters):
                    opposite_parameters = maybe_params_as_unitless(
                        opp_force.getExceptionParameters(opp_exc_idx)[2:])
                    _scales = [_new - _old for _old, _new in zip(orig_nonidx_params,
                                                                opposite_parameters)]
                    new_exc_offset_idx = U_static.addExceptionParameterOffset(
                        exception_offset, orig_exc_idx, *_scales)

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
            _scales = [-1*_i for _i in maybe_params_as_unitless(old_params)]
            _ = U_static.addParticleParameterOffset(
                particle_offset, hybrid_idx, *_scales)
        try:
            new_particle_idx = hybrid_to_new_map[hybrid_idx]
        except:
            new_particle_idx = -1
        if new_particle_idx >=0: # it is mapped;
            # decide whether to add offset.
            new_params = new_nbf.getParticleParameters(new_particle_idx)
            _old_params = maybe_params_as_unitless(old_params)
            _new_params = maybe_params_as_unitless(new_params)
            _scales = [_new - _old for _old, _new in
                zip(_old_params, _new_params)]
            if not np.allclose(_old_params, _new_params):
                _ = U_static.addParticleParameterOffset(
                    particle_offset, hybrid_idx, *_scales)
            else: # old params are same as new params for particle
                # we have already added the old params, so pass
                pass

        else:
            assert old_idx in unique_old_atoms

    for new_idx in unique_new_atoms:
        new_params = new_nbf.getParticleParameters(new_idx)
        hybrid_idx = new_to_hybrid_map[new_idx]
        _ = U_static.addParticleParameterOffset(
            particle_offset, hybrid_idx, *new_params)

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
                         **unused_kwargs) -> unit.Quantity:
    """get hybrid positions from old/new positions; `openmm`-amenable;
    `mapped_positions_on_old` will write mapped positions to the old particle indices;
    otherwise, will map to new."""
    hybrid_positions = np.zeros((num_hybrid_particles, 3))
    old_pos_sans_units = old_positions.value_in_unit_system(unit.md_unit_system)
    new_pos_sans_units = new_positions.value_in_unit_system(unit.md_unit_system)
    to_hybrid_maps = [old_to_hybrid_map, new_to_hybrid_map]
    from_position_cache = [old_pos_sans_units, new_pos_sans_units]

    for to_hybrid_map, from_positions in zip(to_hybrid_maps, from_position_cache):
        for orig_idx, hybrid_idx in to_hybrid_map.items():
            hybrid_positions[hybrid_idx,:] = from_positions[orig_idx,:]
    return hybrid_positions * unit.nanometers

def get_original_positions_from_hybrid(hybrid_positions: unit.Quantity,
                                       hybrid_to_original_map: Dict[int,int],
                                       **unused_kwargs):
    out_positions = np.zeros((len(hybrid_to_original_map), 3))
    hybrid_posits_sans_units = hybrid_positions/unit.nanometer
    for hybrid_idx, orig_idx in hybrid_to_original_map.items():
        out_positions[orig_idx,:] = hybrid_posits_sans_units[hybrid_idx, :]
    return out_positions*unit.nanometer

def energy_by_force(system: openmm.System,
                    reference_positions: unit.Quantity,
                    context_args: Tuple[Any],
                    box_vectors: Tuple[unit.Quantity]=None,
                    global_parameters: Dict[str, int]=None,
                    forces_too: bool=False) -> Dict[str, unit.Quantity]:
    """
    from a unique-force-style system with reference positions/box_vectors,
    iterate through each force object and return the potential energy per
    force object.
    """
    for idx, force in enumerate(system.getForces()):
        force.setForceGroup(idx)

    context = openmm.Context(system, openmm.VerletIntegrator(1.), *context_args)
    if box_vectors is None:
        box_vectors = system.getDefaultPeriodicBoxVectors()
    context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(reference_positions)
    if global_parameters is not None:
        for _name, _val in global_parameters.items():
            context.setParameter(_name, _val)
    out_energies = {}
    out_forces = {}
    for idx, force in enumerate(system.getForces()):
        state = context.getState(getEnergy=True, getForces=True, groups={idx})
        _e = state.getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
        _f = state.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)
        out_energies[force] = _e
        out_forces[force] = _f
    del context
    if forces_too:
        return out_energies, out_forces
    else:
        return out_energies


def getParameters_to_dict(mapstringdouble):
    out_dict = {}
    for key in mapstringdouble:
        out_dict[key] = mapstringdouble[key]
    return out_dict

def setParameters(parameters, context):
    for key, val in parameters.items():
        context.setParameter(key,val)

# class definition

class BaseSingleTopologyHybridSystemFactory(object):
    """
    base class for generating a hybrid system object
    """
    _allowed_force_names = ['HarmonicBondForce',
        'HarmonicAngleForce', 'PeriodicTorsionForce',
        'NonbondedForce', 'MonteCarloBarostat']

    def __init__(
        self: Any,
        old_system: openmm.System, # old_system
        new_system: openmm.System, # new system
        old_to_new_atom_map: Dict[int, int],
        unique_old_atoms: Iterable[int],
        unique_new_atoms: Iterable[int],
        **kwargs):

        self._old_system = copy.deepcopy(old_system)
        self._new_system = copy.deepcopy(new_system)
        self._old_to_new_atom_map = copy.deepcopy(old_to_new_atom_map)
        self._unique_old_atoms = copy.deepcopy(unique_old_atoms)
        self._unique_new_atoms = copy.deepcopy(unique_new_atoms)
        self._hybrid_system = openmm.System()

        # now call setup fns
        self._add_particles_to_hybrid(**kwargs)
        self._get_force_dicts(**kwargs) # render force dicts for new/old sys
        self._assert_allowable_forces(**kwargs)
        self._copy_box_vectors(**kwargs) # copy box vectors
        self._handle_constraints(**kwargs)
        self._handle_virtual_sites(**kwargs)

        self._equip_hybrid_forces(**kwargs)

        if self._hybrid_system.usesPeriodicBoundaryConditions():
            self._copy_barostat(**kwargs) # copy barostat

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

    def _copy_box_vectors(self, **unused_kwargs):
        box_vectors = self._old_system.getDefaultPeriodicBoxVectors()
        self._hybrid_system.setDefaultPeriodicBoxVectors(*box_vectors)

    def _copy_barostat(self, **unused_kwargs):
        if "MonteCarloBarostat" in self._old_forces.keys():
            barostat = copy.deepcopy(self._old_forces["MonteCarloBarostat"])
            self._hybrid_system.addForce(barostat)

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

    def _equip_hybrid_forces(self, **kwargs):
        pass

    @property
    def hybrid_system(self):
        return copy.deepcopy(self._hybrid_system)

class V1SingleTopologyHybridSystemFactory(
    BaseSingleTopologyHybridSystemFactory):
    ATM_EXPR_TEMPLATE = [
    'U0 + (({atm_lambda2} - {atm_lambda1})/{atm_alpha}) * log_term + {atm_lambda2}*u_sc + {atm_w0};',
    'log_term = c + log(exp(-c) + exp(exponand-c));',
    'c = max(0, exponand);',
    'exponand = -{atm_alpha}*(u_sc - {atm_u0});',
    'u_sc = select(soften_bool, u_soft, u);',
    'soften_bool = select(1-{atm_soften_switch}, 0, step(u - {atm_u_cut}));',
    'u_soft = ({atm_u_max} - {atm_u_cut}) * f_sc + {atm_u_cut};',
    'f_sc = (z_y^{atm_a} - 1)/(z_y^{atm_a} + 1);',
    'z_y = 1 + 2*y_by_a + 2*(y_by_a^2);',
    'y_by_a = y / {atm_a};',
    'y = (u - {atm_u_cut}) / ({atm_u_max} - {atm_u_cut});',
    'u = U1 - U0;',
    'U1 = select({atm_leg}, U0_static, U1_static);',
    'U0 = select({atm_leg}, U1_static, U0_static);']

    ATM_EXPR = ''.join(ATM_EXPR_TEMPLATE)
    ATM_COLLECTIVE_VARS = ['U0_static', 'U1_static']
    ATM_GLOBAL_PARAMS = {
    'atm_time': 0.,
    'atm_lambda1': 0.,
    'atm_lambda2': 0.,
    'atm_alpha': 0.1,
    'atm_u_cut': 200., # check this again in comparison to `_u0`
    'atm_u0': 100.,
    'atm_u_max': 400.,
    'atm_a': 0.0625,
    'atm_w0': 0.,
    'atm_soften_switch': 1., # bool to determine whether to allow for soft u_sc,
    'atm_leg': 0.,
    }

    VALENCE_EXPR_TEMPLATE = [
    "U0_static = old_term + static_term + unique_term;",
    "U1_static = new_term + static_term + unique_term;",
    "unique_term = unique_old_switch*unique_old_term + unique_new_switch*unique_new_term;"
    "old_term = select(1-old, 0., U_valence);",
    "new_term = select(1-new, 0., U_valence);",
    "unique_old_term = select(1-uold, 0., U_valence);",
    "unique_new_term = select(1-unew, 0., U_valence);"
    "static_term = select(1-static, 0., U_valence);"]

    VALENCE_EXPR = ' '.join(VALENCE_EXPR_TEMPLATE)
    VALENCE_GLOBAL_PARAMETERS = {'unique_old_switch': 1,
        'unique_new_switch': 1}
    VALENCE_FORCE_UTILS = {
        'HarmonicBondForce': {
            'addTerm': 'addBond',
            'setTerm': 'setBondParameters',
            'query_num_terms': 'getNumBonds',
            'query_params': 'getBondParameters',
            'custom_force': openmm.CustomBondForce,
            'num_particles': 2,
            'add_per_term_param': 'addPerBondParameter',
            'num_params': 2,
            'per_term_params': ['length', 'k', 'static', 'old', 'new', 'uold', 'unew'],
            'U_valence': "U_valence = 0.5*k*(r-length)^2;"},
        'HarmonicAngleForce': {
            'addTerm': 'addAngle',
            'setTerm': 'setAngleParameters',
            'query_num_terms': 'getNumAngles',
            'query_params': 'getAngleParameters',
            'custom_force': openmm.CustomAngleForce,
            'num_particles': 3,
            'add_per_term_param': 'addPerAngleParameter',
            'num_params': 2,
            'per_term_params': ['angle', 'k', 'static', 'old', 'new', 'uold', 'unew'],
            "U_valence": "U_valence = 0.5*k*(theta-angle)^2;"},
        'PeriodicTorsionForce': {
            'addTerm': 'addTorsion',
            'setTerm': 'setTorsionParameters',
            'query_num_terms': 'getNumTorsions',
            'query_params': 'getTorsionParameters',
            'custom_force': openmm.CustomTorsionForce,
            'num_particles': 4,
            'add_per_term_param': 'addPerTorsionParameter',
            'num_parameters': 3,
            'per_term_params': ['periodicity', 'phase', 'k', 'static', 'old', 'new', 'uold', 'unew'],
            "U_valence": "U_valence = k*(1 + cos(periodicity*theta - phase));"}
        }

    NONBONDED_GLOBAL_PARAMETERS = {'unique_old_exception_switch': 1.,
        'unique_new_exception_switch': 1.,
        'exception_offset' : 0.,
        'particle_offset': 0.}
    def __init__(self,
        *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _default_protocol(self, time, parameters) -> Dict[str, float]:
        """from a set of parameters, given the time, update them accordingly;
        `time` runs from 0 to 1"""
        lambda1 = time if time < 0.5 else 1. - time
        lambda2 = lambda1
        leg = 0 if time < 0.5 else 1
        updater = {
            'atm_time': time,
            'atm_lambda1': lambda1,
            'atm_lambda2': lambda2,
            'atm_leg': leg,
            }
        parameters.update(updater)
        return parameters


    def _equip_hybrid_forces(self, **kwargs):
        """
        equip the valence and nonbonded forces.
        """
        # manage the global params/custom expr first
        atm_global_param_format = {key: key for key in self.ATM_GLOBAL_PARAMS.keys()}
        atm_base_expression = copy.deepcopy(self.ATM_EXPR).format(**atm_global_param_format)

        # get the set of union forcenames
        joint_forcenames = set(list(self._old_forces.keys()) + list(self._new_forces.keys()))

        valence_custom_expr = atm_base_expression + self.VALENCE_EXPR
        valence_forcenames = list(self.VALENCE_FORCE_UTILS.keys())
        valence_global_params = copy.deepcopy(self.ATM_GLOBAL_PARAMS)
        valence_global_params.update(self.VALENCE_GLOBAL_PARAMETERS)
        for forcename in joint_forcenames:
            # warning, this will fail if the valence force is not in both systems.
            if forcename in valence_forcenames: # if it is valence
                old_force = self._old_forces[forcename]
                new_force = self._new_forces[forcename]
                out_force = translate_standard_valence_force_to_custom(
                    old_force = old_force,
                    new_force = new_force,
                    old_to_hybrid_map = self._old_to_hybrid_map,
                    new_to_hybrid_map = self._new_to_hybrid_map,
                    unique_old_atoms = self._unique_old_atoms,
                    unique_new_atoms = self._unique_new_atoms,
                    force_utils = self.VALENCE_FORCE_UTILS[forcename],
                    custom_expression = valence_custom_expr,
                    global_parameters = valence_global_params,
                    static_del = [1, 0, 0, 0, 0],
                    old_del = [0, 1, 0, 0, 0],
                    new_del = [0, 0, 1, 0, 0],
                    unique_old_del = [0, 0, 0, 1, 0],
                    unique_new_del = [0, 0, 0, 0, 1], **kwargs)
                _ = self._hybrid_system.addForce(out_force)

        # now nonbonded
        if 'NonbondedForce' in joint_forcenames:
            nonbonded_global_param_names = {key: key for
                key in self.NONBONDED_GLOBAL_PARAMETERS.keys()}
            u0_nb, u1_nb = translate_standard_nonbonded_force(
                old_nbf=self._old_forces['NonbondedForce'],
                new_nbf=self._new_forces['NonbondedForce'],
                num_hybrid_particles = self._hybrid_system.getNumParticles(),
                old_to_hybrid_map = self._old_to_hybrid_map,
                new_to_hybrid_map = self._new_to_hybrid_map,
                unique_old_atoms = self._unique_old_atoms,
                unique_new_atoms = self._unique_new_atoms,
                **nonbonded_global_param_names)
            cv = openmm.CustomCVForce(atm_base_expression)
            for coll_var_name, coll_var in zip(self.ATM_COLLECTIVE_VARS, [u0_nb, u1_nb]):
                cv.addCollectiveVariable(coll_var_name, copy.deepcopy(coll_var))
            for param_name, param_value in self.ATM_GLOBAL_PARAMS.items():
                cv.addGlobalParameter(param_name, param_value)
            _ = self._hybrid_system.addForce(cv)

    def test_energy_endstates(self, old_positions, new_positions,
        atol=1e-1, rtol=1e-6, verbose=False,
        context_args = (), **unused_kwargs):
        hybrid_system = self.hybrid_system # get the hybrid system

        # make match querier dict.
        std_to_custom_forcename = {key: _dict['custom_force']('').__class__.__name__
            for key, _dict in self.VALENCE_FORCE_UTILS.items()}
        std_to_custom_forcename['NonbondedForce'] = 'CustomCVForce' # handle nonbonded
        std_to_custom_forcename['MonteCarloBarostat'] = 'MonteCarloBarostat'

        # get positions.
        hybrid_positions = get_hybrid_positions(old_positions=old_positions,
            new_positions = new_positions, num_hybrid_particles=hybrid_system.getNumParticles(),
            old_to_hybrid_map = self._old_to_hybrid_map, new_to_hybrid_map = self._new_to_hybrid_map)
        old_positions = get_original_positions_from_hybrid(hybrid_positions = hybrid_positions,
            hybrid_to_original_map = self._hybrid_to_old_map)
        new_positions = get_original_positions_from_hybrid(hybrid_positions = hybrid_positions,
            hybrid_to_original_map = self._hybrid_to_new_map)

        # first, retrieve the old/new system forces by object
        if verbose: print(f"computing old force energies.")
        old_force_dict = energy_by_force(system = copy.deepcopy(self._old_system),
            reference_positions = old_positions, context_args = context_args)
        if verbose: print(f"computing new force energies")
        new_force_dict = energy_by_force(system = copy.deepcopy(self._new_system),
            reference_positions = new_positions, context_args = context_args)

        # now set the valence global parameters to match old
        zero_atm_global_params = {'atm_time': 0.,
            'atm_leg': 0.,
            'unique_old_switch': 1.,
            'unique_new_switch': 0.,
            'U0_static_unique_old_exception_switch': 1.,
            'U0_static_unique_new_exception_switch': 0.,
            'U1_static_unique_old_exception_switch': 1.,
            'U1_static_unique_new_exception_switch': 0.}
        switch_lambda = lambda x: 1. if x==0. else 0.
        one_atm_global_params = {key: switch_lambda(val) for key, val in zero_atm_global_params.items()}
        if verbose: print(f"computing old hybrid force energies...")
        hybr_old_force_dict = energy_by_force(system = self.hybrid_system,
            reference_positions = hybrid_positions, global_parameters=zero_atm_global_params,
            context_args = context_args)
        if verbose: print(f"computing new hybrid force energies...")
        hybr_new_force_dict = energy_by_force(system = copy.deepcopy(self.hybrid_system),
            reference_positions = hybrid_positions, global_parameters=one_atm_global_params,
            context_args = context_args)

        original_joint_forcenames = set(list(self._old_forces.keys())).union(
            set(list(self._new_forces.keys())))

        for original_forcename in original_joint_forcenames:
            custom_forcename = std_to_custom_forcename[original_forcename]
            old_energy = old_force_dict[original_forcename]
            new_energy = new_force_dict[original_forcename]
            if original_forcename == 'MonteCarloBarostat':
                if verbose: print(f"omitting `MonteCarloBarostat`")
            else:
                hybrid_old_energy = hybr_old_force_dict[custom_forcename]
                hybrid_new_energy = hybr_new_force_dict[custom_forcename]
                for endstate, orig_e, hybr_e in zip(['old', 'new'],
                    [old_energy, new_energy], [hybrid_old_energy, hybrid_new_energy]):
                    if not np.isclose(orig_e, hybr_e, atol=atol, rtol=rtol):
                        raise Exception(f"""for original force {original_forcename}
                            (custom force {custom_forcename}) at {endstate} state,
                            energies do not match: {orig_e} vs {hybr_e}, respectively.
                            """)
                    elif verbose:
                        print(f"""original force {original_forcename} with energy {orig_e}
                            at state {endstate} match. ({orig_e} and {hybr_e}, respectively)""")
                    else:
                        pass

class SCRFSingleTopologyHybridSystemFactory(BaseSingleTopologyHybridSystemFactory):
  """
  SoftCore ReactionField Single Topology HybridSystemFactory;
  WARNING: this operation can expect to take ~15s in complex phase.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _default_protocol(self, time, parameters, **unused_kwargs) -> Dict[str, float]:
    updater = {'lambda_global': time}
    parameters.update(updater)
    return parameters

  def _make_rf_systems(self, **kwargs):
    """internal utility to make reaction-field systems
    for energy matching comparisons and reference"""
    from aludel.rf import ReactionFieldConverter
    self._old_rf_system = ReactionFieldConverter(self._old_system, **kwargs).rf_system
    self._new_rf_system = ReactionFieldConverter(self._new_system, **kwargs).rf_system

  def _get_valence_converter(self, **unused_kwargs):
    """get the valence converter factory"""
    from aludel.valence import SingleTopologyHybridValenceConverter
    return SingleTopologyHybridValenceConverter

  def _get_nonbonded_converter(self, **unused_kwargs):
    from aludel.rf import SingleTopologyHybridNBFReactionFieldConverter
    return SingleTopologyHybridNBFReactionFieldConverter

  def _equip_hybrid_forces(self, **kwargs):
    """
    equip the valence and nonbonded forces
    """
    valence_converter = self._get_valence_converter(**kwargs)
    nonbonded_converter = self._get_nonbonded_converter(**kwargs)
    # get the set of union forcenames
    joint_forcenames = set(list(self._old_forces.keys()) + list(self._new_forces.keys()))
    valence_forcenames = [i for i in self._allowed_force_names if i not in ['NonbondedForce',
      'MonteCarloBarostat']]
    for forcename in joint_forcenames:
      if forcename in valence_forcenames: # if it is valence
        # this will fail if the valence force is not in both systems
        old_force = self._old_forces[forcename]
        new_force = self._new_forces[forcename]
        valence_hbf_factory = valence_converter(
          old_force = self._old_forces[forcename],
          new_force = self._new_forces[forcename],
          old_to_hybrid_map = self._old_to_hybrid_map,
          new_to_hybrid_map = self._new_to_hybrid_map,
          num_hybrid_particles = self._hybrid_system.getNumParticles(),
          unique_old_atoms = self._unique_old_atoms,
          unique_new_atoms = self._unique_new_atoms,
          **kwargs)
        out_force = valence_hbf_factory.hybrid_force
        self._hybrid_system.addForce(out_force)

    if 'NonbondedForce' in joint_forcenames:
      nb_converter_factory = nonbonded_converter(
        old_nbf=self._old_forces['NonbondedForce'],
        new_nbf=self._new_forces['NonbondedForce'],
        old_to_hybrid_map=self._old_to_hybrid_map,
        new_to_hybrid_map=self._new_to_hybrid_map,
        num_hybrid_particles=self._hybrid_system.getNumParticles(),
        unique_old_atoms=self._unique_old_atoms,
        unique_new_atoms=self._unique_new_atoms,
        **kwargs)
      hybrid_rf_nbfs = nb_converter_factory.rf_forces
      _ = [self._hybrid_system.addForce(_q) for _q in hybrid_rf_nbfs]

  def test_energy_endstates(self, old_positions, new_positions, atol=1e-2,
    rtol=1e-6, verbose=False, context_args=(),
    old_global_parameters = {'lambda_global': 0., 'retain_exception_switch': 0., 'retain_uniques': 0.},
    new_global_parameters = {'lambda_global': 1., 'retain_exception_switch': 0., 'retain_uniques': 0.},
    return_energy_differences=False, **kwargs):
    """test the endstates energy bookkeeping here;
    WARNING: for complex phase, this is an expensive operation (~30s on CPU).
    """
    from aludel.atm import (energy_by_force,
      get_hybrid_positions, get_original_positions_from_hybrid)
    from aludel.rf import ReactionFieldConverter

    if verbose: print(f"generating old rf system...")
    old_rf_system = ReactionFieldConverter(self._old_system, **kwargs).rf_system
    if verbose: print(f"generating new rf system...")
    new_rf_system = ReactionFieldConverter(self._new_system, **kwargs).rf_system

    hybrid_system = self.hybrid_system
    hybrid_positions = get_hybrid_positions(old_positions=old_positions,
      new_positions=new_positions, num_hybrid_particles=self._hybrid_system.getNumParticles(),
      old_to_hybrid_map = self._old_to_hybrid_map, new_to_hybrid_map = self._new_to_hybrid_map,
      **kwargs)
    old_positions = get_original_positions_from_hybrid(hybrid_positions=hybrid_positions,
      hybrid_to_original_map=self._hybrid_to_old_map, **kwargs)
    new_positions = get_original_positions_from_hybrid(hybrid_positions=hybrid_positions,
      hybrid_to_original_map=self._hybrid_to_new_map, **kwargs)

    # first, retrieve the old/new system forces by object
    if verbose: print(f"computing old force energies.")
    old_es = energy_by_force(system = copy.deepcopy(old_rf_system),
      reference_positions = old_positions, context_args = context_args)
    if verbose: print(f"computing new force energies")
    new_es = energy_by_force(system = copy.deepcopy(new_rf_system),
      reference_positions = new_positions, context_args = context_args)

    if verbose: print(f"computing old hybrid force energies...")
    hybr_old_es = energy_by_force(system = self.hybrid_system,
      reference_positions = hybrid_positions, global_parameters=old_global_parameters,
      context_args = context_args)
    if verbose: print(f"computing new hybrid force energies...")
    hybr_new_es = energy_by_force(system = copy.deepcopy(self.hybrid_system),
      reference_positions = hybrid_positions, global_parameters=new_global_parameters,
      context_args = context_args)

    # old match
    old_es_sum = np.sum(list(old_es.values()))
    hybr_old_es_sum = np.sum(list(hybr_old_es.values()))
    old_pass = np.isclose(old_es_sum, hybr_old_es_sum,
      atol=atol, rtol=rtol)
    if not old_pass:
      print(f"""energy match of old/hybrid-old system failed; printing energy by forces...\n
        \t{old_es_sum}\n\t{hybr_old_es_sum}""")
    else:
      print(f"passed with energy match: {old_es_sum, hybr_old_es_sum}")

    # new match
    new_es_sum = np.sum(list(new_es.values()))
    hybr_new_es_sum = np.sum(list(hybr_new_es.values()))
    new_pass = np.isclose(new_es_sum, hybr_new_es_sum,
      atol=atol, rtol=rtol)
    if not new_pass:
      print(f"""energy match of new/hybrid-new system failed; printing energy by forces...\n
        \t{new_es_sum}\n\t{hybr_new_es_sum}""")
    else:
      print(f"passed with energy match: {new_es_sum, hybr_new_es_sum}")

    if verbose:
      for nonalch_state, alch_state, state_name in zip(
        [old_es, new_es], [hybr_old_es, hybr_new_es], ['old', 'new']):
        print(f"""printing {state_name} nonalch and alch energies by force:""")
        for _force, _energy in nonalch_state.items():
          print(f"\t", _force.__class__.__name__, _energy)
        print("\n")
        for _force, _energy in alch_state.items():
          print(f"\t", _force.__class__.__name__, _energy)

    if return_energy_differences:
      return [[old_es_sum, hybr_old_es_sum], [new_es_sum, hybr_new_es_sum]]
    else:
      if not old_pass: raise Exception(f"old failed")
      if not new_pass: raise Exception(f"new failed")
