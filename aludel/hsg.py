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
    'get_param_particle_indices': 2
    },
'HarmonicAngleForce': {
    'standard_add_term_expr': 'addAngle',
    'query_num_terms_expr': 'getNumAngles',
    'query_params_expr': 'getAngleParameters',
    'standard_force_obj': openmm.HarmonicAngleForce,
    'get_param_particle_indices': 3
    },
'PeriodicTorsionForce': {
    'standard_add_term_expr': 'addTorsion',
    'query_num_terms_expr': 'getNumTorsions',
    'query_params_expr': 'getTorsionParameters',
    'standard_force_obj': openmm.PeriodicTorsionForce,
    'get_param_particle_indices': 4
    }
}


# utilities


def query_indices(indices_to_query: Iterable[int],
                  to_hybrid_map: Dict[int, int],
                  uniques: Iterable[int],
                  cores: Iterable[int]
                  ) -> Tuple[Iterable[int], Iterable[bool], Iterable[bool]]:
    """
    utility fn used primarily for `translate_standard_valence_force` below;
    Arguments:
        indices_to_query: an iterable of integer indices of old/new particle
            indices
        to_hybrid_map: a dict of old/new to hybrid indices
        uniques: an iterable of integer indices of old/new particle indices
        cores: an iterable of integer indices of old/new particles indices

    Returns:
        hybrid_indices: hybrid indices corresponding to `indices_to_query`
        unique_bools: iterable of bools of whether the given indices are
            unique to the system
        core_bools: iterable of bools of whether the given indices are
            core to both systems
    """
    hybrid_indices = [old_to_hybrid_map[_idx] for _idx in indices_to_query]
    unique_bools = [_idx in uniques for _idx in indices_to_query]
    core_bools = [_idx in cores for _idx in indices_to_query]
    return hybrid_indices, unique_bools, core_bools

def translate_standard_valence_force(
    old_force: openmm.Force,
    new_force: openmm.Force,
    old_to_hybrid_map: Dict[int, int],
    new_to_hybrid_map: Dict[int, int],
    unique_news: Iterable[int],
    unique_olds: Iterable[int],
    old_to_new_core_dict: Dict[int, int],
    **kwargs):
    """translate a standard valence_force"""
    old_forcename = old_force.__class__.__name__
    new_forcename = new_force.__class__.__name__
    is_periodic = old_force.usesPeriodicBoundaryConditions() or \
        new_force.usesPeriodicBoundaryConditions()
    assert old_forcename == new_forcename, f"""
    old force {old_forcename} doesn't match new force {new_forcename}
    """
    force_util_dict = VALENCE_FORCE_STATIC_EXPR[old_forcename]
    num_particles_per_term = force_util_dict['get_param_particle_indices']

    U0_out_force = force_dict['standard_force_obj']()
    U1_out_force = force_dict['standard_force_obj']()
    static_out_force = force_dict['standard_force_obj']()

    # periodic consistency
    for new_force in [U0_out_force, U1_out_force, static_out_force]:
        new_force.setUsesPeriodicBoundaryConditions(is_periodic)

    U0_add_term_fn = getattr(U0_out_force,
                             force_dict['standard_add_term_expr'])
    U1_add_term_fn = getattr(U1_out_force,
                             force_dict['standard_add_term_expr'])
    static_add_term_fn = getattr(static_out_force,
                                 force_dict['standard_add_term_expr'])

    query_force_dict = {'old': old_force,
                        'new': new_force}
    for query_force_key, query_force in query_force_dict.items():
        num_terms = getattr(query_force,
                                force_util_dict['query_num_terms_expr'])()
        get_params_fn = getattr(query_force,
                                force_util_dict['query_params_expr'])
        _to_hybrid_map = old_to_hybrid_map if query_force_key=='old' \
            else new_to_hybrid_map
        _unique_atoms = unique_old_atoms if query_force_key=='old' \
            else unique_new_atoms
        _core_atoms = list(self._old_to_new_atom_map.keys()) if \
            query_force_key=='old' else list(self._old_to_new_atom_map.values())
        for term_idx in range(num_terms):
            all_parameters = get_params_fn(term_idx)
            particles = all_parameters[:num_particles_per_term]
            per_term_parameters = all_parameters[num_particles_per_term:]
            hybrid_particles, uniques, cores = query_indices(particles,
                                                             _to_hybrid_map,
                                                             _unique_atoms,
                                                             _core_atoms)
        if sum(uniques) > 0:
            # the term comes exclusively from old/new force; make static
            static_add_term_fn(*hybrid_particles, *per_term_parameters)
        elif sum(cores) == num_particles_per_term:
            # these are cores, so add them to either U0 or U1
            if query_force_key == 'old':
                U0_add_term_fn(*hybrid_particles, *per_term_parameters)
            elif query_force_key == 'new':
                U1_add_term_fn(*hybrid_particles, *per_term_parameters)
            else:
                raise ValueError(f"""
                query force key {query_force_key} is not supported""")
        elif sum(uniques) + sum(cores) == 0:
            # this term is neither unique nor core; must be env; make static
            static_add_term_fn(*hybrid_particles, *per_term_parameters)
        else:
            raise ValueError(f"""
            particles {particles} from {query_force_key} {old_forcename}
            are not behaving;
            their hybrid indices are {hybrid_particles};
            their unique bools look like {uniques};
            their core bools look like {cores};
            """)
    # now we have three force objects. return them
    return static_out_force, U0_out_force, U1_out_force

def translate_nonbonded_delimiters(
    old_force: openmm.NonbondedForce,
    **kwargs):
    new_nbf = openmm.NonbondedForce()
    new_nbf.setNonbondedMethod(old_force.getNonbondedMethod())
    new_nbf.setCutoffDistance(old_force.getCutoffDistance())
    new_nbf.setEwaldErrorTolerance(old_force.getEwaldErrorTolerance())
    new_nbf.setLJPMEParameters(*old_force.getLJPMEParameters())
    new_nbf.setPMEParameters(*old_force.getPMEParameters())
    new_nbf.setUseSwitchingDistance(old_force.getUseSwitchingDistance())
    new_nbf.setSwitchingDistance(old_force.getSwitchingDistance())
    new_nbf.setUseDispersionCorrection(old_force.getUseDispersionCorrection())
    return new_nbf

def translate_standard_nonbonded_force(
    old_nbf: openmm.NonbondedForce,
    new_nbf: openmm.NonbondedForce,
    old_to_hybrid_map: Dict[int, int],
    new_to_hybrid_map: Dict[int, int],
    unique_news: Iterable[int],
    unique_olds: Iterable[int],
    **kwargs) -> Tuple[openmm.NonbondedForce, openmm.NonbondedForce]:
    """
    translate a standard nonbonded force into 2 forces
    Returns:
        out_old_nbf: out `openmm.NonbondedForce` corresponding to `U0_static`
        out_new_nbf: out `openmm.NonbondedForce` corresponding to `U1_static`
    """
    out_old_nbf = translate_nonbonded_delimiters(old_nbf, **kwargs)
    out_new_nbf = translate_nonbonded_delimiters(new_nbf, **kwargs)
    assert all([key==val for key, val in old_to_hybrid_map.items()]), \
        f"""nbf translation requirement for consistency"""
    old_num_particles = old_nbf.getNumParticles()
    for old_idx in range(old_num_particles):
        charge, sigma, epsilon = old_nbf.getParticleParameters(old_idx)
        out_old_nbf_idx = out_old_nbf.addParticle(charge, sigma, epsilon)
        out_new_nbf_idx = out_new_nbf.addParticle(charge, sigma, epsilon)
        is_unique = old_idx in unique_olds
        if is_unique:
            out_new_nbf.setParticleParameter(
                out_new_nbf_idx, charge*0., sigma, epsilon*0.)

    # now get the rest of the hybrid indices (sorted)
    hybr_indices_to_add = sorted([val for val in new_to_hybrid_map.values() if \
        val >= old_num_particles])
    hybrid_to_new_map = {val: key for key, val in new_to_hybrid_map}
    new_num_particles = new_nbf.getNumParticles()
    for new_idx in range(new_num_particles):
        charge, sigma, epsilon = new_nbf.getParticleParameters(new_idx)
        hybrid_idx = hybrid_to_new_map[new_idx]
        if hybrid_idx < old_num_particles: # overwrite previously-defined params
            out_new_nbf.setParticleParameter(hybrid_idx, charge, sigma, epsilon)
        else: # assert it is a unique new
            assert new_idx in unique_news, f"""new nbf idx {new_idx} with
                hybrid idx {hybrid_idx} is not a unique new, but is greater
                than the number of particles in the old nbf {old_num_particles}
                """
    # iterate over the sorted hybrid indices via the new nbf to add unique news
    for hybrid_idx in hybr_indices_to_add:
        new_idx = hybrid_to_new_map[hybrid_idx]
        assert new_idx in unique_news, f"""new idx queried is {new_idx} from
            hybrid idx {hybrid_idx}, but it is not in unique news"""
        charge, sigma, epsilon = new_nbf.getParticleParameters(new_idx)
        _old_idx = out_old_nbf.addParticle(charge*0., sigma, epsilon*0.)
        _new_idx = out_new_nbf.addParticle(charge, sigma, epsilon)
        assert _old_idx == hybrid_idx, f"""added index to old system is
            {_old_idx} but the queried hybrid idx in {hybrid_idx}"""
        assert _old_idx == _new_idx, f"particles should be added contiguously"

    # now exceptions; exceptions are _always_ on.
    for nbf_key, nbf in zip(['old', 'new'], [old_nbf, new_nbf]):
        to_out_force = out_old_nbf if nbf_key == 'old' else out_new_nbf
        num_exceptions = nbf.getNumExceptions()
        _to_hybrid_map = old_to_hybrid_map if nbf_key == 'old' else \
            new_to_hybrid_map
        for idx in range(num_exceptions):
            p1, p2, chargep, sigma, eps = nbf.getExceptionParameters(idx)
            hybrid_p1, hybrid_p2 = _to_hybrid_map[p1], _to_hybrid_map[p2]
            to_out_force.addException(hybrid_p1, hybrid_p2, chargep, sigma, eps)
    return out_old_nbf, out_new_nbf

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

        self._old_to_hybrid_map = {}
        self._new_to_hybrid_map = {}

        self._atm_expression_template = atm_expression_template
        self._atm_collective_variable_names = atm_collective_variables
        self._atm_default_global_parameters = atm_default_global_parameters

        # now call setup fns
        self._get_force_dicts(**kwargs) # render force dicts for new/old sys
        self._assert_allowable_forces(**kwargs)
        self._copy_barostat(**kwargs) # copy barostat
        self._copy_box_vectors(**kwargs) # copy box vectors
        self._add_particles_to_hybrid(**kwargs) # create old/new to hybrid dicts
        self._get_hybrid_to_maps(**kwargs) # flip old/new to hybrid dicts

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

    def _get_hybrid_to_maps(self, **unused_kwargs):
        """from the completed `{old/new}_to_hybrid_map` maps, create
        `hybrid_to_{new/old}_map`"""
        self._hybrid_to_old_map = {value:key for key, val in \
            self._old_to_hybrid_map.items()}
        self._hybrid_to_new_map = {value:key for key, val in \
            self._new_to_hybrid_map.items()}

    def _add_particles_to_hybrid(self, **unused_kwargs):
        """copy all old system particles to hybrid with identity mapping"""
        old_num_particles = self._old_system.getNumParticles()
        for idx in range(old_num_particles): # iterate over old particles
            mass_old = self._old_system.getParticleMass(idx)
            if idx in self._old_to_new_atom_map.keys():
                idx_in_new_sys = self._old_to_new_atom_map[idx]
                mass_new = self._new_system.getParticleMass(idx)
                mix_mass = (mass_old + mass_new) / 2
            else:
                mix_mass = mass
            hybrid_idx = self._hybrid_system.addParticle(mix_mass)
            self._old_to_hybrid_map[idx] = hybrid_idx

        for idx in self._unique_new_atoms: # iterate over new particles
            mass = self._new_system.getParticleMass(idx)
            hybrid_idx = self._hybrid_system.addParticle(mass)
            self._new_to_hybrid_map[idx] = hybrid_idx

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
                static_f, U0_f, U1_f = translate_standard_valence_force(
                    old_force=old_force,
                    new_force=new_force,
                    old_to_hybrid_map=self._old_to_hybrid_map,
                    new_to_hybrid_map=self._new_to_hybrid_map,
                    unique_news=self._unique_new_atoms,
                    unique_olds=self._unique_old_atoms,
                    **kwargs)
                out_force_dict[force_name] = {
                    'static': static_f,
                    'U0_static': U0_f,
                    'U1_static': U1_f}
        return out_force_dict

    def _make_nonbonded_force(self,
        nonbondedforce_name: str='NonbondedForce',
        **kwargs):
        """construct U0 and U1 nonbondedforces to equip to a CustomCVForce
        by calling `translate_standard_nonbonded_force`
        """
        nbf_dict = {'old': None, 'new': None}
        for nbf_gather_key in nbf_dict.keys():
            query_forces = getattr(self, f"_{nbf_gather_key}_forces")
            query_forcenames = query_forces.keys()
            if nonbondedforce_name in query_keys:
                nbf_dict[nbf_gather_key] = query_forces[nonbondedforce_name]
            else:
                raise Exception(f"_{nbf_gather_key}_forces does not contain \
                forcename {nonbondedforce_name}")
        U0_nbf, U1_nbf = translate_standard_nonbonded_force(
            old_nbf = nbf_dict['old'],
            new_nbf = nbf_dict['new'],
            old_to_hybrid_map = self._old_to_hybrid_map,
            new_to_hybrid_map = self._new_to_hybrid_map,
            unique_news = self._unique_new_atoms,
            unique_olds = self._unique_old_atoms,
            **kwargs)
        return U0_nbf, U1_nbf

    def _build_hybrid_system_forces(self, **kwargs):
        """
        This function constructs the finalized `hybrid_system` by calling
        `_make_valence_forces` and `_make_nonbonded_force` and equipping the
        force objects into a `CustomCVForce`.

        NOTE: in this construction, the
        """
        offset_default_global_parameters = {key[1:] : value for key, value in
            self._atm_default_global_parameters} # remove `_` in param names
        energy_expr = self._atm_expression_template.format(
        **offset_default_global_parameters) # register params in place
        coll_vars = {_key: valence_force_dict[_key] for key in
            self._atm_collective_variable_names} # make coll vars dict
        # valence forces
        valence_force_dict = _make_valence_forces(**kwargs)
        for valence_force_name, valence_force_dict in valence_force.items():
            # first, equip the static valence force
            self._hybrid_system.addForce(
                copy.deepcopy(valence_force_dict['static']))

            # then construct and equip CustomCVForce
            cv = make_CustomCVForce(
                global_parameters = self._atm_default_global_parameters,
                energy_fn_expression = energy_expr,
                energy_fn_collective_vars = coll_vars,
                **kwargs)
            self._hybrid_system.addForce(cv)

        # NonbondedForce
        U0_nbf, U1_nbf = _make_nonbonded_force(**kwargs)
        nbf_cv = make_CustomCVForce(
            global_parameters = self._atm_default_global_parameters,
            energy_fn_expression = energy_expr,
            energy_fn_collective_vars = coll_vars,
            **kwargs)
        self._hybrid_system.addForce(nbf_cv)

    @property
    def hybrid_system(self):
        return copy.deepcopy(self._hybrid_system)
