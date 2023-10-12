"""utility functions and classes to parameterize `openmm` systems/topologies from datafiles;
also contains some map-rendering functionality"""

# Imports
import openmm
import mdtraj as md
import numpy as np

from typing import Callable, Dict, Iterable, Tuple, Any, List
from rdkit import Chem
from openff.toolkit.topology import Molecule
from openmm import unit, app
from copy import deepcopy


def Molecule_from_sdf(sdf, **unused_kwargs):
    """make an openff Molecule from an sdf"""
    with open(sdf, mode='rb') as file:
        sdf_object = Molecule.from_file(file, file_format="SDF")
    return sdf_object


def omm_solvate_with_padding(omm_topology: openmm.app.Topology,
                             omm_positions: unit.Quantity,
                             addSolvent_kwargs: Dict) -> Tuple[openmm.app.Topology, unit.Quantity]:
    """return a tuple of solvated topology and positions"""
    modeller = openmm.app.Modeller(omm_topology, omm_positions)
    modeller.addSolvent(**addSolvent_kwargs)
    solvated_omm_topology = modeller.getTopology()
    _solvated_omm_positions = modeller.getPositions()
    solvated_omm_positions = unit.Quantity(value=np.array([list(atom_pos) for atom_pos
                                                           in _solvated_omm_positions.value_in_unit_system(
            unit.md_unit_system)]), unit=unit.nanometers)
    return solvated_omm_topology, solvated_omm_positions


def make_new_omm_top(old_omm_topology: openmm.app.Topology,
                     new_resname_omm_topology: openmm.app.Topology,
                     resname: str = 'MOL', **unused_kwargs):
    """given an `old` fully-prepped topology and a resname target make the `new` omm topology"""
    md_topology_old = md.Topology.from_openmm(old_omm_topology)  # convert old prepped top to md
    res_stripped_md_top = md_topology_old.subset(md_topology_old.select(f"not resname {resname}"))  # strip the resname

    # join the res-stripped top to new resname top
    new_joined_md_top = md.Topology.from_openmm(new_resname_omm_topology).join(res_stripped_md_top)
    # hack the output md topology to a dataframe because bugs
    nsl, b = new_joined_md_top.to_dataframe()
    out_md_topology = md.Topology.from_dataframe(nsl, b)
    out_omm_topology = out_md_topology.to_openmm()

    out_omm_topology.setPeriodicBoxVectors(old_omm_topology.getPeriodicBoxVectors())  # set pbcs if not none
    return out_omm_topology


def find_res_indices(omm_topology: openmm.app.Topology, resname: str = 'MOL'):
    """find the start index of an `openmm.app.Topology` object that matches a specific resname and the length thereof"""
    res_atom_indices = []
    for res in omm_topology.residues():
        if res.name == resname:  # found the match
            for atom in res.atoms():
                res_atom_indices.append(atom.index)
    if len(res_atom_indices) == 0:
        raise Exception(f"there is an inconsistency; res atoms is empty")
    return res_atom_indices


def compute_distance_matrix(p1: np.array, p2: np.array) -> np.array:
    """compute the distance between two position arrays"""
    import jax
    from jax import numpy as jnp
    d = lambda a, b: jnp.linalg.norm(a - b)
    vdis = jax.vmap(d, in_axes=(None, 0))
    vvdis = jax.vmap(vdis, in_axes=(0,None))
    return vvdis(p1, p2)

def new_to_old_distance_mapper(old_mol_positions: np.array,
                               new_mol_positions: np.array,
                               distance_threshold: float = 0.02, **unused_kwargs) -> Dict[int, int]:
    """a means of creating a v1 `new_to_old` atom map where the map is defined by a distance threshold
    between any two atoms. it warrants inspection for scaffold hops"""
    distance_matrix = compute_distance_matrix(old_mol_positions, new_mol_positions)
    threshold_bool = distance_matrix < distance_threshold  # make a boolean threshold
    old_to_new_matches = np.argwhere(threshold_bool)[::-1]  # reverse

    # now we just need to reduce duplicates...
    # note that selection between any two entries is a toss-up
    out_match_dict = {}
    for entry in old_to_new_matches:
        _old, _new = entry
        out_match_dict[_new] = _old

    return out_match_dict


def repair_constraint_mapping(
        new_to_old_map: Dict[int, int],
        unique_olds: List[int],
        unique_news: List[int],
        old_ligand_system: openmm.System,
        new_ligand_system: openmm.System,
        old_ligand_topology: openmm.app.Topology,
        new_ligand_topology: openmm.app.Topology,
        **unused_kwargs):
    """if the system is parameterized using HBond constraints,
    it is necessary to 'repair' the initial atom map after parameterization
    by demapping hydrogen atoms where the constraint of the bond changes"""

    # make a copy of the original new to old atom map
    out_new_to_old_atom_map = deepcopy(new_to_old_map)
    out_unique_olds = deepcopy(unique_olds)
    out_unique_news = deepcopy(unique_news)

    # get old/new hydrogens
    old_hs = [atom.index for atom in old_ligand_topology.atoms() if atom.element == app.Element.getByAtomicNumber(1)]
    new_hs = [atom.index for atom in new_ligand_topology.atoms() if atom.element == app.Element.getByAtomicNumber(1)]

    # make an internal function to query the hydrogen constraints
    def _make_constraint_dict(_system, hs_indices):
        _dict = {}
        for idx in range(_system.getNumConstraints()):
            a1, a2, _len = _system.getConstraintParameters(idx)
            if a1 in hs_indices: _dict[a1] = _len
            if a2 in hs_indices: _dict[a2] = _len
        return _dict

    old_constraint_dict = _make_constraint_dict(old_ligand_system, old_hs)
    new_constraint_dict = _make_constraint_dict(new_ligand_system, new_hs)

    to_del = []
    for new_idx, old_idx in new_to_old_map.items():
        if new_idx in new_constraint_dict.keys() and old_idx in old_constraint_dict.keys():
            old_len, new_len = old_constraint_dict[old_idx], new_constraint_dict[new_idx]
            if not np.isclose(old_len.value_in_unit_system(unit.md_unit_system),
                              new_len.value_in_unit_system(unit.md_unit_system)):
                # if the constraint distances don't match, delete it
                to_del.append(new_idx)
        elif old_idx in old_constraint_dict.keys() and new_idx not in new_constraint_dict.keys():
            to_del.append(new_idx)
        elif old_idx not in old_constraint_dict.keys() and new_idx in new_constraint_dict.keys():
            to_del.append(new_idx)

    for idx in to_del:  # indices are `new`
        old_idx = new_to_old_map[idx]  # query the old member
        del out_new_to_old_atom_map[idx]
        out_unique_olds.append(old_idx)
        out_unique_news.append(idx)

    return out_new_to_old_atom_map, out_unique_olds, out_unique_news, to_del


def get_full_atom_map_and_uniques(
        ligand_new_to_old_map: Dict[int, int],
        old_omm_topology: openmm.app.Topology,
        new_omm_topology: openmm.app.Topology,
        resname: str = 'MOL',
        **unused_kwargs
):
    """given a ligand-only new-to-old atom map (zero-indexed),
    figure out how to expand the map to accommodate mapped `environment` atoms in
    old/new omm_topologies; also return the offset value."""
    old_num_residues = old_omm_topology.getNumResidues()
    new_num_residues = new_omm_topology.getNumResidues()
    assert old_num_residues == new_num_residues, f"""
        it is assumed that the old/new topologies have the same number of residues"""

    old_omm_residues = list(old_omm_topology.residues())
    new_omm_residues = list(new_omm_topology.residues())

    new_to_old_atom_map = {}
    for residx in range(len(old_omm_residues)):
        old_res, new_res = old_omm_residues[residx], new_omm_residues[residx]
        assert old_res.name == new_res.name, f"old res name does not match new res name"
        if old_res.name != resname:  # handle mapping of conserved residues
            for old_atom, new_atom in zip(old_res.atoms(), new_res.atoms()):
                assert old_atom.name == new_atom.name
                new_to_old_atom_map[new_atom.index] = old_atom.index
        else:
            old_indices = [atom.index for atom in old_res.atoms()]
            new_indices = [atom.index for atom in new_res.atoms()]
            assert old_indices[0] == new_indices[0]
            offset_lig_new_to_old_map = {key + old_indices[0]: val + old_indices[0]
                                         for key, val in ligand_new_to_old_map.items()}
            new_to_old_atom_map.update(offset_lig_new_to_old_map)

    # cool, get the unique atoms
    unique_news = [atom for atom in new_indices if atom not in offset_lig_new_to_old_map.keys()]
    unique_olds = [atom for atom in old_indices if atom not in offset_lig_new_to_old_map.values()]

    # get the offset value
    ligand_offset_value = old_indices[0]

    return new_to_old_atom_map, ligand_offset_value, unique_olds, unique_news


def render_mol_map(
        mol_old: Chem.Mol,
        mol_new: Chem.Mol,
        new_to_old_atom_map: Dict[int, int],
        align_substructures: bool = True,
        MolsToGridImage_kwargs: Dict = {'subImgSize': (600, 600)},
        **unused_kwargs):
    """
    make a mol mapping for two small molecules for visual mapping purposes in `jupyter.notebook`
    """
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.drawOptions.addAtomIndices = True
    from rdkit import Geometry
    from rdkit.Chem import AllChem, Draw

    match = np.transpose(np.array([[j, i] for i, j in new_to_old_atom_map.items()])).tolist()
    if align_substructures:  # if align substructures for 2d rendering
        AllChem.Compute2DCoords(mol_old)
        coords = [mol_old.GetConformer().GetAtomPosition(x) for x in match[0]]
        coords2d = [Geometry.Point2D(pt.x, pt.y) for pt in coords]

        coord_dict = {match[1][i]: coord for i, coord in enumerate(coords2d)}
        AllChem.Compute2DCoords(mol_new, coordMap=coord_dict)

    img = Draw.MolsToGridImage([mol_old, mol_new],
                               molsPerRow=2, highlightAtomLists=match, **MolsToGridImage_kwargs)

    return img


def fix_barostat_in_place(system: openmm.System, barostat_name: str = 'MonteCarloBarostat', **unused_kwargs):
    """remove a `MonteCarloBarostat` if the system is not periodic;
    not sure why this is happening, exactly"""
    for idx, force in enumerate(system.getForces()):
        if force.__class__.__name__ == barostat_name:
            system.removeForce(idx)
            break
    return system


class RenderSolventOMMObjects(object):
    """A simple handler class to parameterize ligands/complexes into `openmm`
    """

    def __init__(self,
                 ligand_old: Molecule, 
                 ligand_new: Molecule, 
                 SystemGenerator_kwargs: Dict[str, Any] = {
                     'forcefields': ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
                     'small_molecule_forcefield': 'openff-1.3.0',
                     'barostat': openmm.MonteCarloBarostat(1. * unit.atmosphere, 300 * unit.kelvin, 50),
                     'forcefield_kwargs': {'removeCMMotion': False,
                                           'ewaldErrorTolerance': 2.5e-4,
                                           'constraints': app.HBonds,
                                           'hydrogenMass': 3.5 * unit.amus,
                                           'rigidWater': True},
                     'periodic_forcefield_kwargs': {'nonbondedMethod': app.PME},
                     'nonperiodic_forcefield_kwargs': {'nonbondedMethod': app.NoCutoff}},
                 addSolvent_kwargs: Dict[str, Any] = {
                     'model': 'tip3p',
                     'padding': 15. * unit.angstroms,
                     'ionicStrength': 0.15 * unit.molar},
                 ):
        from openmmforcefields.generators import SystemGenerator

        # make openff.toolkit `molecules` from sdf files
        self._Molecule_old = ligand_old
        self._Molecule_new = ligand_new

        # make system gen first since it's used for all phases/endstates
        self._system_generator = SystemGenerator(
            molecules=[self._Molecule_old, self._Molecule_new],
            **SystemGenerator_kwargs)

        # update the `addSolvent` kwargs to update the ff used from `SystemGenerator`
        addSolvent_kwargs.update({'forcefield': self._system_generator.forcefield})
        self._addSolvent_kwargs = addSolvent_kwargs

        # reassign name of mol to target
        for _mol in [self._Molecule_old, self._Molecule_new]:
            _ = _mol.name = 'MOL'

        # make vacuum positions, systems, topologies (old/new); make vacuum by default
        self._make_vacuum_pos_sys_top()

    def _make_vacuum_pos_sys_top(self, **unused_kwargs):
        """in-place generator for old/new vacuum positions, system, topology"""
        # old/new top
        self._vacuum_omm_topology_old = self._Molecule_old.to_topology().to_openmm()
        self._vacuum_omm_topology_new = self._Molecule_new.to_topology().to_openmm()

        # old/new pos; these go to angstrom for some reason
        self._vacuum_omm_positions_old = self._Molecule_old._conformers[0].to_openmm()
        self._vacuum_omm_positions_new = self._Molecule_new._conformers[0].to_openmm()

        # make systems
        self._vacuum_omm_system_old = fix_barostat_in_place(
            self._system_generator.create_system(self._vacuum_omm_topology_old))

        self._vacuum_omm_system_new = fix_barostat_in_place(
            self._system_generator.create_system(self._vacuum_omm_topology_new))

    def _make_solvent_pos_sys_top(self, **unused_kwargs):
        # make solvent old topology/positions
        self._solvent_omm_topology_old, self._solvent_omm_positions_old = omm_solvate_with_padding(
            omm_topology=self._vacuum_omm_topology_old,
            omm_positions=self._vacuum_omm_positions_old,
            addSolvent_kwargs=self._addSolvent_kwargs)

        # the solvent new topology, positions are rendered slightly differently; we use the mapping for this
        self._solvent_omm_topology_new = make_new_omm_top(old_omm_topology=self._solvent_omm_topology_old,
                                                          new_resname_omm_topology=self._vacuum_omm_topology_new)
        self._solvent_omm_positions_new = self._make_solvent_omm_positions_new()

        # make solvent system
        self._solvent_omm_system_old = self._system_generator.create_system(self._solvent_omm_topology_old)
        self._solvent_omm_system_new = self._system_generator.create_system(self._solvent_omm_topology_new)

    def _make_solvent_omm_positions_new(self):
        """make the new positions; this might be a bit sketch"""
        num_new_atoms = self._solvent_omm_topology_new.getNumAtoms()
        vac_posits_new = self._vacuum_omm_positions_new.value_in_unit(unit.nanometer)
        solvent_posits_old = self._solvent_omm_positions_old.value_in_unit(unit.nanometer)
        solvent_omm_positions_new = np.zeros((num_new_atoms, 3))

        for old_res, new_res in zip(self._solvent_omm_topology_old.residues(),
                                    self._solvent_omm_topology_new.residues()):
            old_indices = [atom.index for atom in old_res.atoms()]
            new_indices = [atom.index for atom in new_res.atoms()]
            if old_res.name == 'MOL':
                assert new_res.name == 'MOL'
                solvent_omm_positions_new[new_indices, :] = vac_posits_new
            else:
                solvent_omm_positions_new[new_indices, :] = solvent_posits_old[old_indices, :]

        return solvent_omm_positions_new * unit.nanometer
