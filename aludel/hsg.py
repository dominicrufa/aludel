"""utility functions for generating hybrid systems; let's rename this `.py`;
it makes me sad"""

import openmm
from openmm import app, unit
import numpy
import copy
from typing import Any, Tuple, Dict, Iterable, Callable

# define basic alchemical transfer expression
V1_ATM_EXPR_TEMPLATE = """
(({_lambda2} - {_lambda1})/{_alpha}) * log_term + {_lambda2}*u_sc + {_w0};
log_term = log_c + log(exp(0-c) + exp(exponand-c);
c = max(0, exponand);
exponand = -{_alpha}*(u_sc - {_u0});
u_sc = select(soften_bool, u_soft, u);
soften_bool = step(u - {_u_cut});
u_soft = ({_u_max} - {_u_cut}) * f_sc + {_u_cut};
f_sc = (z_y^{_a} - 1)/(z_y^{_a} + 1);
z_y = 1 + 2*y_by_a + 2*(y_by_a^2);
y_by_a = y / {_a};
y = (u - {_u_cut}) / ({_u_max} - {_u_cut});
u = U1 - U0;
U1 = select(past_half, U0_static, U1_static);
U0 = select(past_half, U1_static, U0_static);
past_half = step({_time} - 0.5);
"""
V1_ATM_COLLECTIVE_VARS = ['U0_static', 'U1_static']
V1_ATM_DEFAULT_GLOBAL_PARAMS = {
'_time': 0.,
'_lambda1': 0.,
'_lambda2': 0.,
'_alpha': 0.1,
'_u_cut': 110., # check this again in comparison to `_u0`
'_u0': 100.,
'_u_max': 200.,
'_a': 0.0625,
'_w0': 0.,
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
    },
'HarmonicAngleForce': {
    'standard_add_term_expr': 'addAngle',
    'query_num_terms_expr': 'getNumAngles',
    'query_params_expr': 'getAngleParameters',
    'standard_force_obj': openmm.HarmonicAngleForce,
    'get_param_particle_indices': 3,
    'parameter_setter_fn': 'setAngleParameters'
    },
'PeriodicTorsionForce': {
    'standard_add_term_expr': 'addTorsion',
    'query_num_terms_expr': 'getNumTorsions',
    'query_params_expr': 'getTorsionParameters',
    'standard_force_obj': openmm.PeriodicTorsionForce,
    'get_param_particle_indices': 4,
    'parameter_setter_fn': 'setTorsionParameters',
    }
}


# utilities
def query_indices(indices_to_query: Iterable[int],
                  to_hybrid_map: Dict[int, int],
                  uniques: Iterable[int],
                  mapped: Iterable[int]
                  ) -> Tuple[Iterable[int], Iterable[bool], Iterable[bool]]:
    """
    utility fn used primarily for `translate_standard_valence_force` below;
    Arguments:
        indices_to_query: an iterable of integer indices of old/new particle
            indices
        to_hybrid_map: a dict of old/new to hybrid indices
        uniques: an iterable of integer indices of old/new particle indices that are not mapped
        mapped: an iterable of integer indices of old/new particles indices that are mapped

    Returns:
        hybrid_indices: hybrid indices corresponding to `indices_to_query`
        unique_bools: iterable of bools of whether the given indices are
            unique to the system
        mapped_bools: iterable of bools of whether the given indices are
            mapped to both systems
    """
    hybrid_indices = [to_hybrid_map[_idx] for _idx in indices_to_query]
    unique_bools = [_idx in uniques for _idx in indices_to_query]
    mapped_bools = [_idx in mapped for _idx in indices_to_query]
    return hybrid_indices, unique_bools, mapped_bools

def are_static_valence_terms(terms_1: Iterable,
                             terms_2: Iterable,
                             **unused_kwargs) -> bool:
    """given an iterable of valence terms (either floats, unit'd term, or int),
       return whether all terms are static
    """
    are_static = True
    for _term1, _term2 in zip(terms_1, terms_2):
        term1_type, term2_type = type(_term1), type(_term2)
        assert term1_type == term2_type, f"""
        term 1 is of type {term1_type} but term 2 is of
        type {term2_type}"""
        if type(_term1) in [float, int]:
            if not np.isclose(_term1, _term2):
                are_static = False
                break
        elif type(_term1) == openmm.unit.quantity.Quantity:
            assert _term1.unit == _term2.unit, f"""
            term 1 has units of {_term1.unit} but term 2 has units of
            {_term2.unit}"""
            if not np.isclose(_term1.value_in_unit_system(unit.md_unit_system),
                             _term2.value_in_unit_system(unit.md_unit_system)):
                are_static = False
                break
    return are_static

def translate_standard_valence_force(old_force: openmm.Force,
                                     new_force: openmm.Force,
                                     old_to_hybrid_map: Iterable[int],
                                     new_to_hybrid_map: Iterable[int],
                                     unique_old_atoms: Iterable[int],
                                     unique_new_atoms: Iterable[int],
                                     **kwargs):
    """translate a standard valence force"""
    old_forcename = old_force.__class__.__name__
    new_forcename = new_force.__class__.__name__
    unique_term_indices = {'old': [], 'new': []} # bookkeep unique terms

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
    # opposite force
    for unique_label, term_indices in unique_term_indices.items():
        out_force = new_force if unique_label=='old' else old_force
        query_force = old_force if unique_label=='old' else new_force
        write_force = U1_force if unique_label=='old' else U0_force
        query_params_fn = getattr(query_force,
                        force_util_dict['query_params_expr'])
        to_hybrid_map = old_to_hybrid_map if unique_label == 'old' else new_to_hybrid_map
        term_add_fn = getattr(write_force,
                              force_util_dict['standard_add_term_expr'])
        for term_idx in term_indices:
            all_parameters = query_params_fn(term_idx)
            particles = all_parameters[:num_particles_per_term]
            per_term_parameters = all_parameters[num_particles_per_term:]
            hybrid_indices = [to_hybrid_map[_idx] for _idx in particles]
            term_add_fn(*hybrid_indices, *per_term_parameters)

    return U0_force, U1_force, unique_term_indices

def translate_standard_nonbonded_force(
    old_nbf: openmm.NonbondedForce,
    new_nbf: openmm.NonbondedForce,
    num_hybrid_particles: int,
    old_to_hybrid_map: Dict[int, int],
    new_to_hybrid_map: Dict[int, int],
    unique_new_atoms: Iterable[int],
    unique_old_atoms: Iterable[int],
    treat_exceptions_like_valence: bool=True,
    **kwargs) -> Tuple[openmm.NonbondedForce, openmm.NonbondedForce]:
    """
    translate a standard nonbonded force
    """
    U0_static = copy.deepcopy(old_nbf)
    U1_static = copy.deepcopy(new_nbf)
    num_old_particles = old_nbf.getNumParticles()
    num_new_particles = new_nbf.getNumParticles()
    assert all([key==val for key, val in old_to_hybrid_map.items()]), \
        f"""nbf translation requirement for consistency"""
    assert num_old_particles == len(old_to_hybrid_map)

    hybrid_to_old_map = {val:key for key, val in old_to_hybrid_map.items()}
    hybrid_to_new_map = {val:key for key, val in new_to_hybrid_map.items()}

    # handle particle parameters

    # add appropriate number of unique particles
    for _ in range(len(unique_old_atoms)):
        _ = U1_static.addParticle(0., 0., 0.)
    assert U1_static.getNumParticles() == num_hybrid_particles
    for _ in range(len(unique_new_atoms)):
        _ = U0_static.addParticle(0., 0., 0.)
    assert U0_static.getNumParticles() == num_hybrid_particles

    # iterate over all old/hybrid indices
    for old_idx, hybrid_idx in old_to_hybrid_map.items():
        # this is redundant because of first assert statement
        old_params = old_nbf.getParticleParameters(old_idx)
        U0_static.setParticleParameters(hybrid_idx, *old_params)
    for new_idx, hybrid_idx in new_to_hybrid_map.items():
        new_params = new_nbf.getParticleParameters(new_idx)
        U1_static.setParticleParameters(hybrid_idx, *new_params)

    # iterate over particle exception parameters;
    # do we need the same number of exceptions in each?
    static_exception_indices = {'U0_static': [], 'U1_static': []}
    for force, to_hybrid_map in zip([old_nbf, new_nbf],
        [old_to_hybrid_map, new_to_hybrid_map]):
        num_exceptions = force.getNumExceptions()
        for exception_idx in range(num_exceptions):
            p1, p2, cp, s, e = force.getExceptionParameters(exception_idx)
            hybr_p1, hybr_p2 = to_hybrid_map[p1], to_hybrid_map[p2]
            force.setExceptionParameters(exception_idx, hybr_p1, hybr_p2,
                                        cp, s, e)

            if treat_exceptions_like_valence:
                write_to_force = U1_static if force == old_nbf \
                    else U0_static
                write_to_key = 'U1_static' if force == old_nbf \
                    else 'U0_static'
                uniques = unique_old_atoms if force == old_nbf else \
                    unique_new_atoms
                includes_unique = any([_p in uniques for _p in [p1, p2]])
                if includes_unique:
                    static_exception_idx = write_to_force.addException(
                        hybr_p1, hybr_p2, cp, s, e, replace=True)
                    static_exception_indices[write_to_key].append(
                        static_exception_idx)
    return U0_static, U1_static, static_exception_indices



def make_CustomCVForce(
    global_parameters: Dict[str, float],
    energy_fn_expression: str,
    energy_fn_collective_vars: Dict[float, openmm.Force],
    **unused_kwargs):
    cv = openmm.CustomCVForce('')
    for param_name, param_value in global_parameters.items():
        cv.addGlobalParameter(param_name, param_value)
    for energy_fn, force in energy_fn_collective_vars.items():
        cv.addCollectiveVariable(energy_fn, copy.deepcopy(force))
    cv.setEnergyFunction(energy_fn_expression)
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
        atm_expression_template: str=V1_ATM_EXPR_TEMPLATE,
        atm_collective_variable_names: Iterable[str]=V1_ATM_COLLECTIVE_VARS,
        atm_default_global_parameters: Dict[str, float]=V1_ATM_DEFAULT_GLOBAL_PARAMS,
        **kwargs):

        self._old_system = copy.deepcopy(old_system)
        self._new_system = copy.deepcopy(new_system)

        self._old_to_new_atom_map = copy.deepcopy(old_to_new_atom_map)

        self._unique_old_atoms = copy.deepcopy(unique_old_atoms)
        self._unique_new_atoms = copy.deepcopy(unique_new_atoms)

        self._hybrid_system = openmm.System()

        self._atm_expression_template = atm_expression_template
        self._atm_collective_variable_names = atm_collective_variable_names
        self._atm_default_global_parameters = atm_default_global_parameters

        # now call setup fns
        self._add_particles_to_hybrid(**kwargs)
        self._get_force_dicts(**kwargs) # render force dicts for new/old sys
        self._assert_allowable_forces(**kwargs)
        self._copy_barostat(**kwargs) # copy barostat
        self._copy_box_vectors(**kwargs) # copy box vectors
        self._handle_constraints(**kwargs)
        self._handle_virtual_sites(**kwargs)
        self._equip_valence_forces(**kwargs)
#         self._equip_nonbonded_force(**kwargs)


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
        for idx in range(old_num_particles): # iterate over old particles
            mass_old = self._old_system.getParticleMass(idx)
            if idx in self._old_to_new_atom_map.keys():
                idx_in_new_sys = self._old_to_new_atom_map[idx]
                mass_new = self._new_system.getParticleMass(idx)
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
                U0_f, U1_f, static_term_dict = translate_standard_valence_force(
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
                    'static_term_dict': static_term_dict}
        return out_force_dict

    def _equip_valence_forces(self, **kwargs):
        """make `openmm.CustomCVForce` for valence forces and equip
        to `self._hybrid_system`"""
        offset_default_global_parameter_names = {key : key[1:] for key in
            self._atm_default_global_parameters.keys()} # remove `_` in param names
        energy_expr = self._atm_expression_template.format(
            **offset_default_global_parameter_names) # register params in place

        # valence forces
        valence_force_dict = self._make_valence_forces(**kwargs)
        self._static_valence_force_terms = {} # for bookkeeping
        for valence_force_name, valence_force_terms in valence_force_dict.items():
            # equip the static valence force terms for bookkeeping
            self._static_valence_force_terms[valence_force_name] = \
                valence_force_terms['static_term_dict']

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
        offset_default_global_parameter_names = {key : key[1:] for key in
            self._atm_default_global_parameters.keys()} # remove `_` in param names
        energy_expr = self._atm_expression_template.format(
            **offset_default_global_parameter_names) # register params in place

        # NonbondedForce
        old_nbf = self._old_forces[nonbonded_force_name]
        new_nbf = self._new_forces[nonbonded_force_name]
        U0_nbf, U1_nbf, static_exception_indices = \
            translate_standard_nonbonded_force(
            old_nbf = old_nbf,
            new_nbf = new_nbf,
            num_hybrid_particles = self._hybrid_system.getNumParticles(),
            old_to_hybrid_map = self._old_to_hybrid_map,
            new_to_hybrid_map = self._new_to_hybrid_map,
            unique_new_atoms = self._unique_new_atoms,
            unique_old_atoms = self._unique_old_atoms,
            **kwargs)
        # then construct and equip CustomCVForce
        coll_vars = {'U0_static': U0_nbf, 'U1_static': U1_nbf}
        nbf_cv = make_CustomCVForce(
            global_parameters = self._atm_default_global_parameters,
            energy_fn_expression = energy_expr,
            energy_fn_collective_vars = coll_vars,
            **kwargs)
        self._static_nonbonded_exception_indices = static_exception_indices
        self._hybrid_system.addForce(nbf_cv)

    @property
    def hybrid_system(self):
        return copy.deepcopy(self._hybrid_system)
