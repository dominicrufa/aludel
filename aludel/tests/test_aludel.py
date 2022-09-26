"""
Unit and regression test for the aludel package.
"""
import openmm
from openmm import app, unit
import numpy
import copy
import numpy as np
from typing import Any, Tuple, Dict, Iterable, Callable
# from aludel import hsg
from aludel.utils import decompress_pickle
from aludel.hsg import *

DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-6
def energy_by_force(system: openmm.System,
                    reference_positions: unit.Quantity,
                    box_vectors: Tuple[unit.Quantity]=None,

                    **unused_kwargs) -> Dict[str, unit.Quantity]:
    """
    from a unique-force-style system with reference positions/box_vectors,
    iterate through each force object and return the potential energy per
    force object.
    """
    forcename_to_index = {}
    forcename_to_energy = {}
    for idx, force in enumerate(system.getForces()):
        force_name = force.__class__.__name__
        force.setForceGroup(idx)
        if force_name in forcename_to_index.keys():
            raise Exception(f"""force name {force_name} already exists in
            the system. This breaks the assumption on the construction of
            the `openmm.Force` object""")
        forcename_to_index[force_name] = idx

    context = openmm.Context(system, openmm.VerletIntegrator(1.))
    if box_vectors is None:
        box_vectors = system.getDefaultPeriodicBoxVectors()
    context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(reference_positions)
    for force_name, idx in forcename_to_index.items():
        state = context.getState(getEnergy=True, groups={idx})
        _e = state.getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
        forcename_to_energy[force_name] = _e
    del context
    return forcename_to_energy

def extract_energies_from_CustomCVForce(context,
                                        customcvforce,
                                        **unused_kwargs):
    """from a context containing a `CustomCVForce`, (and aforementioned
    `CustomCVForce`), return the energy of each collective variable"""
    num_cvs = customcvforce.getNumCollectiveVariables()
    cv_energies = customcvforce.getCollectiveVariableValues(context)
    cv_name_to_energy = {customcvforce.getCollectiveVariableName(idx):
                        cv_energies[idx] for idx in range(num_cvs)
                        }
    return cv_name_to_energy

def mod_cv_nonbonded_parameters(
    customcvforce,
    **unused_kwargs):
    """
    modify the customcvForce collective vars in place
    """
    num_cvs = customcvforce.getNumCollectiveVariables()
    cv_name_to_force = {
        customcvforce.getCollectiveVariableName(idx) :
        customcvforce.getCollectiveVariable(idx)
        for idx in range(num_cvs)}
    force_names = [val.__class__.__name__ for val
                   in cv_name_to_force.values()]
    assert len(set(force_names)) == 1, f"""
    all force names in cvforce ({force_names}) must be identical
    """
    # for `U0_static`, set the unique new exc param (idx=1) to 0.
    cv_name_to_force['U0_static'].setGlobalParameterDefaultValue(
        1, 0.)
    cv_name_to_force['U1_static'].setGlobalParameterDefaultValue(
        0, 0.)

def mod_cv_valence_parameters(customcvforce,
                              artificial_term_dict,
                             **unused_kwargs):
    """
    we want to modify the customcvForce collective vars in place
    with artificial terms.
    """
    num_cvs = customcvforce.getNumCollectiveVariables()
    artificial_term_name_set = set(list(artificial_term_dict.keys()))
    cv_name_to_force = {
        customcvforce.getCollectiveVariableName(idx) :
        customcvforce.getCollectiveVariable(idx)
        for idx in range(num_cvs)}
    force_names = [val.__class__.__name__ for val
                   in cv_name_to_force.values()]
    assert len(set(force_names)) == 1, f"""
    all force names in cvforce ({force_names}) must be identical
    """
    force_util_dict = VALENCE_FORCE_STATIC_EXPR[force_names[0]]
    num_particles = force_util_dict['get_param_particle_indices']
    cv_names = cv_name_to_force.keys()
    assert artificial_term_name_set == set(cv_names), f"""
        the corresponding names of the artificial term name set
        {artificial_term_name_set} is not equal to the cv names {cv_names}
        """
    for cv_name, cv in cv_name_to_force.items():
        artificial_indices = artificial_term_dict[cv_name]
        term_setter_fn = getattr(cv, force_util_dict['parameter_setter_fn'])
        term_getter_fn = getattr(cv, force_util_dict['query_params_expr'])
        for term_idx in artificial_indices:
            all_parameters = term_getter_fn(term_idx)
            particles = all_parameters[:num_particles]
            params = all_parameters[num_particles:]
            mod_params = [0*_param for _param in params]
            if force_names[0] == 'PeriodicTorsionForce':
                mod_params[0] = 1
            term_setter_fn(term_idx, *particles, *mod_params)

def get_original_positions_from_hybrid(hybrid_positions: unit.Quantity,
                                       hybrid_to_original_map: Dict[int,int],
                                       **unused_kwargs):
    out_positions = np.zeros((len(hybrid_to_original_map), 3))
    hybrid_posits_sans_units = hybrid_positions/unit.nanometer
    for hybrid_idx, orig_idx in hybrid_to_original_map.items():
        out_positions[orig_idx,:] = hybrid_posits_sans_units[hybrid_idx, :]
    return out_positions*unit.nanometer


def test_BaseHybridSystemFactory(
    bhsf,
    old_positions,
    new_positions,
    box_vectors=None,
    **kwargs):

    # make hybrid/old/new positions
    hybrid_positions = get_hybrid_positions(old_positions, new_positions,
                         num_hybrid_particles=bhsf.hybrid_system.getNumParticles(),
                         old_to_hybrid_map=bhsf._old_to_hybrid_map,
                         new_to_hybrid_map=bhsf._new_to_hybrid_map,
                         mapped_positions_on_old=False,
                         **kwargs)
    old_from_hybrid_positions = get_original_positions_from_hybrid(
        hybrid_positions=hybrid_positions,
        hybrid_to_original_map=bhsf._hybrid_to_old_map,
        **kwargs)
    new_from_hybrid_positions = get_original_positions_from_hybrid(
        hybrid_positions=hybrid_positions,
        hybrid_to_original_map=bhsf._hybrid_to_new_map,
        **kwargs)


    # get old/new system potential energies by forcename
    old_system_energies = energy_by_force(bhsf.old_system,
                                          old_from_hybrid_positions,
                                          **kwargs)
    new_system_energies = energy_by_force(bhsf.new_system,
                                          new_from_hybrid_positions,
                                          **kwargs)

    # modify the hybrid system to recover unmodded params
    hybrid_system = bhsf.hybrid_system
    for force in hybrid_system.getForces():
        force_name = force.__class__.__name__
        if force_name == 'CustomCVForce':
            inner_force_name = force.getCollectiveVariable(0).__class__.__name__
            if inner_force_name == 'NonbondedForce':
                mod_cv_nonbonded_parameters(
                    force, **kwargs)
            elif inner_force_name in list(VALENCE_FORCE_STATIC_EXPR.keys()):
                artificial_term_dict = bhsf._artificial_valence_force_terms[inner_force_name]
                mod_cv_valence_parameters(force, artificial_term_dict)
            else:
                raise Exception(f"blah")

    context = openmm.Context(hybrid_system, openmm.VerletIntegrator(1.))
    if box_vectors is None:
        box_vectors = hybrid_system.getDefaultPeriodicBoxVectors()
    context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(hybrid_positions)

    mod_hybrid_energies = {}
    for force in hybrid_system.getForces():
        force_name = force.__class__.__name__
        if force_name == 'CustomCVForce':
            inner_force_name = force.getCollectiveVariable(0).__class__.__name__
            cv_energies = extract_energies_from_CustomCVForce(context,
                                        force)
            mod_hybrid_energies[inner_force_name] = cv_energies


    # now for the assertion part
    old_set_forcenames = set(list(old_system_energies.keys())).union(
        list(new_system_energies.keys()))
    hybrid_cv_forcenames = set(list(mod_hybrid_energies.keys()))
    assert hybrid_cv_forcenames.issubset(old_set_forcenames), f"""
    the cv-collected hybrid forces is not a subset of the old/new forces
    ({old_set_forcenames})"""

    for force_name, energy_dict in mod_hybrid_energies.items():
        for hybrid_static_name, hybrid_static_energy in energy_dict.items():
            query_orig_energy_dict = old_system_energies if \
                hybrid_static_name=='U0_static' else \
                new_system_energies
            query_orig_key = 'old' if hybrid_static_name=='U0_static'\
                else 'new'
            query_energy = query_orig_energy_dict[force_name]
        assert np.isclose(
            hybrid_static_energy, query_energy,
            atol=DEFAULT_ATOL,
            rtol=DEFAULT_RTOL,
                         ), f"""
            energy match of forcename={force_name} in hybrid
            {hybrid_static_name} ({hybrid_static_energy}) does
            not match {query_orig_key} of {query_energy}"""
