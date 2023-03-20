import openmm
from openmm import app, unit
import numpy as np
import copy
from typing import Any, Tuple, Dict, Iterable, Callable
from aludel.utils import maybe_params_as_unitless, sort_indices_to_str


# utilities


def get_hybrid_positions(old_positions: unit.Quantity,
                         new_positions: unit.Quantity,
                         num_hybrid_particles: int,
                         old_to_hybrid_map: Dict[int, int],
                         new_to_hybrid_map: Dict[int, int],
                         mapped_indices_to: str = 'old',
                         **unused_kwargs) -> unit.Quantity:
    """get hybrid positions from old/new positions; `openmm`-amenable;
    indices in the `old/new_to_hybrid_map` may both have unique positions;
    `mapped_positions_to` defines from which endstate mapped position will be defined"""
    hybrid_positions = np.zeros((num_hybrid_particles, 3))
    old_pos_sans_units = old_positions.value_in_unit_system(unit.md_unit_system)
    new_pos_sans_units = new_positions.value_in_unit_system(unit.md_unit_system)
    to_hybrid_maps = [old_to_hybrid_map, new_to_hybrid_map]
    from_position_cache = [old_pos_sans_units, new_pos_sans_units]

    # flip the old/new iteration depending on `mapped_indices_to`
    to_hybrid_maps = to_hybrid_maps[::-1] if mapped_indices_to == 'old' else to_hybrid_maps
    from_position_cache = from_position_cache[::-1] if mapped_indices_to == 'old' else from_position_cache

    for to_hybrid_map, from_positions in zip(to_hybrid_maps, from_position_cache):
        for orig_idx, hybrid_idx in to_hybrid_map.items():
            hybrid_positions[hybrid_idx, :] = from_positions[orig_idx, :]
    return hybrid_positions * unit.nanometers


def get_original_positions_from_hybrid(hybrid_positions: unit.Quantity,
                                       hybrid_to_original_map: Dict[int, int],
                                       **unused_kwargs):
    """get the old/new positions from hybrid"""
    out_positions = np.zeros((len(hybrid_to_original_map), 3))
    hybrid_posits_sans_units = hybrid_positions / unit.nanometer
    for hybrid_idx, orig_idx in hybrid_to_original_map.items():
        out_positions[orig_idx, :] = hybrid_posits_sans_units[hybrid_idx, :]
    return out_positions * unit.nanometer


def energy_by_force(system: openmm.System,
                    reference_positions: unit.Quantity,
                    context_args: Tuple[Any],
                    box_vectors: Tuple[unit.Quantity] = None,
                    global_parameters: Dict[str, int] = None,
                    forces_too: bool = False) -> Dict[str, unit.Quantity]:
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
        context.setParameter(key, val)


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
            old_system: openmm.System,  # old_system
            new_system: openmm.System,  # new system
            old_to_new_atom_map: Dict[int, int],
            unique_old_atoms: Iterable[int],
            unique_new_atoms: Iterable[int],
            **kwargs):

        self._old_system = copy.deepcopy(old_system)
        self._new_system = copy.deepcopy(new_system)
        self._old_to_new_atom_map = copy.deepcopy(old_to_new_atom_map)
        self._unique_old_atoms = copy.deepcopy(unique_old_atoms)
        self._unique_new_atoms = copy.deepcopy(unique_new_atoms)

        # create the hybrid system
        self._hybrid_system = openmm.System()

        # create a dict of hybrid forces
        self._hybrid_forces_dict = {}

        # now call setup fns
        self._add_particles_to_hybrid(**kwargs)
        self._get_force_dicts(**kwargs)  # render force dicts for new/old sys
        self._assert_allowable_forces(**kwargs)
        self._copy_box_vectors(**kwargs)  # copy box vectors
        self._handle_constraints(**kwargs)
        self._handle_virtual_sites(**kwargs)

        if self._hybrid_system.usesPeriodicBoundaryConditions():
            self._copy_barostat(**kwargs)  # copy barostat

        self._equip_hybrid_forces(**kwargs)

    def _add_particles_to_hybrid(self, **unused_kwargs):
        """copy all old system particles to hybrid with identity mapping"""
        self._old_to_hybrid_map = {}
        self._new_to_hybrid_map = {}

        old_num_particles = self._old_system.getNumParticles()
        new_num_particles = self._new_system.getNumParticles()
        for idx in range(old_num_particles):  # iterate over old particles
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
        assert {key == val for key, val in self._old_to_hybrid_map.items()}, f"""
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

        for idx in self._unique_new_atoms:  # iterate over new particles
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
        """iterate over the `self._hybrid_forces_dict` (nested) and add the forces to the `hybrid_system`"""
        duplicate_hybrid_system = copy.deepcopy(self._hybrid_system)
        for forcename, nested_dict in self._hybrid_forces_dict.items():
            out_forces = [copy.deepcopy(x) for x in list(nested_dict.values())]
            _ = [duplicate_hybrid_system.addForce(x) for x in out_forces]

        return duplicate_hybrid_system



class SCRFSingleTopologyHybridSystemFactory(BaseSingleTopologyHybridSystemFactory):
    """
    SoftCore ReactionField Single Topology HybridSystemFactory;
    WARNING: this operation can expect to take ~15s in complex phase.
    """

    def __init__(self, *args, omission_sets: List[Set[int]]=None, **kwargs):
        self._omission_sets = omission_sets
        super().__init__(*args, **kwargs)

    def _equip_hybrid_forces(self, **kwargs):
        """equip the valence and nonbonded forces. Also create a `self._force_factories` dict that
        retains the data generated by each force object"""
        valence_converter = self._get_valence_converter(**kwargs)
        nonbonded_converter = self._get_nonbonded_converter(**kwargs)

        # get the set of union forcenames
        joint_forcenames = set(list(self._old_forces.keys()) + list(self._new_forces.keys()))
        valence_forcenames = [i for i in self._allowed_force_names if i not in ['NonbondedForce',
                                                                                'MonteCarloBarostat']]

        self._force_factories = {}
        for forcename in joint_forcenames:
            if forcename in valence_forcenames:  # if it is valence
                # this will fail if the valence force is not in both systems
                old_force = self._old_forces[forcename]
                new_force = self._new_forces[forcename]
                valence_hbf_factory = valence_converter(
                    old_force=self._old_forces[forcename],
                    new_force=self._new_forces[forcename],
                    old_to_hybrid_map=self._old_to_hybrid_map,
                    new_to_hybrid_map=self._new_to_hybrid_map,
                    num_hybrid_particles=self._hybrid_system.getNumParticles(),
                    unique_old_atoms=self._unique_old_atoms,
                    unique_new_atoms=self._unique_new_atoms,
                    omission_sets=self._omission_sets
                    **kwargs)
                out_force_dict = valence_hbf_factory.hybrid_forces  # `static` and `dynamic` keys by default
                self._hybrid_forces_dict.update(out_force_dict)
                self._force_factories[forcename] = valence_hbf_factory

        if 'NonbondedForce' in joint_forcenames:
            nb_converter_factory = nonbonded_converter(
                old_nbf=self._old_forces['NonbondedForce'],
                new_nbf=self._new_forces['NonbondedForce'],
                old_to_hybrid_map=self._old_to_hybrid_map,
                new_to_hybrid_map=self._new_to_hybrid_map,
                num_hybrid_particles=self._hybrid_system.getNumParticles(),
                unique_old_atoms=self._unique_old_atoms,
                unique_new_atoms=self._unique_new_atoms,
                omission_sets=self._omission_sets
                **kwargs)
            hybrid_rf_nbfs = nb_converter_factory.rf_forces
            self._hybrid_forces_dict.update(hybrid_rf_nbfs)
            self._force_factories[forcename] = nb_converter_factory

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

    def test_energy_endstates(self, old_positions, new_positions, atol=1e-2,
                              rtol=1e-6, verbose=False, context_args=(),
                              old_global_parameters: Dict[str, float]={'lambda_global': 0., 'retain_uniques': 0.},
                              new_global_parameters: Dict[str, float]={'lambda_global': 1., 'retain_uniques': 0.},
                              **kwargs):
        """test the endstates energy bookkeeping here;
        WARNING: for complex phase, this is an expensive operation (~30s on CPU).
        """
        from aludel.atm import energy_by_force, get_hybrid_positions, get_original_positions_from_hybrid
        from aludel.rf import ReactionFieldConverter

        if verbose: print(f"generating old rf system; may take time...")
        old_rf_system = ReactionFieldConverter(self._old_system, **kwargs).rf_system
        if verbose: print(f"generating new rf system; may take time...")
        new_rf_system = ReactionFieldConverter(self._new_system, **kwargs).rf_system

        hybrid_system = self.hybrid_system
        hybrid_positions = get_hybrid_positions(old_positions=old_positions,
                                                new_positions=new_positions,
                                                num_hybrid_particles=self._hybrid_system.getNumParticles(),
                                                old_to_hybrid_map=self._old_to_hybrid_map,
                                                new_to_hybrid_map=self._new_to_hybrid_map,
                                                **kwargs)
        old_positions = get_original_positions_from_hybrid(hybrid_positions=hybrid_positions,
                                                           hybrid_to_original_map=self._hybrid_to_old_map, **kwargs)
        new_positions = get_original_positions_from_hybrid(hybrid_positions=hybrid_positions,
                                                           hybrid_to_original_map=self._hybrid_to_new_map, **kwargs)

        # first, retrieve the old/new system forces by object
        if verbose: print(f"computing old force energies.")
        old_es = energy_by_force(system=copy.deepcopy(old_rf_system),
                                 reference_positions=old_positions, context_args=context_args)
        if verbose: print(f"computing new force energies")
        new_es = energy_by_force(system=copy.deepcopy(new_rf_system),
                                 reference_positions=new_positions, context_args=context_args)

        if verbose: print(f"computing old hybrid force energies...")
        hybr_old_es = energy_by_force(system=self.hybrid_system,
                                      reference_positions=hybrid_positions, global_parameters=old_global_parameters,
                                      context_args=context_args)
        if verbose: print(f"computing new hybrid force energies...")
        hybr_new_es = energy_by_force(system=self.hybrid_system,
                                      reference_positions=hybrid_positions, global_parameters=new_global_parameters,
                                      context_args=context_args)

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
        new_pass = np.isclose(new_es_sum, hybr_new_es_sum, atol=atol, rtol=rtol)
        if not new_pass:
            print(f"""energy match of new/hybrid-new system failed; printing energy by forces...\n
                \t{new_es_sum}\n\t{hybr_new_es_sum}""")
        else:
            print(f"passed with energy match: {new_es_sum, hybr_new_es_sum}")

        if verbose:
            for nonalch_state, alch_state, state_name in \
                    zip([old_es, new_es], [hybr_old_es, hybr_new_es], ['old', 'new']):
                print(f"""printing {state_name} nonalch and alch energies by force:""")
                for _force, _energy in nonalch_state.items():
                    print(f"\t", _force.__class__.__name__, _energy)
                print("\n")
                for _force, _energy in alch_state.items():
                    print(f"\t", _force.__class__.__name__, _energy)

        return [[old_es_sum, hybr_old_es_sum], [new_es_sum, hybr_new_es_sum]]
