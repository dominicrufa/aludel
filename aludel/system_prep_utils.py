"""utility functions and classes to parameterize `openmm` systems/topologies from datafiles;
also contains some map-rendering functionality"""

# Imports
import openmm
import mdtraj as md
import numpy as np

from typing import Callable, Dict, Iterable, Tuple, Any
from rdkit import Chem
from openff.toolkit.topology import Molecule
from openmm import unit, app


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


def new_to_old_distance_mapper(old_mol_positions: np.array,
                               new_mol_positions: np.array,
                               distance_threshold: float = 0.02, **unused_kwargs) -> Dict[int, int]:
    """a means of creating a v1 `new_to_old` atom map where the map is defined by a distance threshold
    between any two atoms. it warrants inspection for scaffold hops"""
    import jax
    import jax_md

    disp, _ = jax_md.space.free()
    # make an N_old x N_new matrix of distances
    distance_matrix = jax_md.space.map_product(
        metric_or_displacement=jax_md.space.canonicalize_displacement_or_metric(disp))(new_mol_positions,
                                                                                       old_mol_positions)
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
        old_ligand_system: openmm.System,
        new_ligand_system: openmm.System,
        old_ligand_topology: openmm.app.Topology,
        new_ligand_topology: openmm.app.Topology,
        **unused_kwargs):
    """if the system is parameterized using HBond constraints,
    it is necessary to 'repair' the initial atom map after parameterization
    by demapping hydrogen atoms where the constraint of the bond changes"""
    from copy import deepcopy
    out_new_to_old_atom_map = deepcopy(new_to_old)

    # get old/new hydrogems
    old_hs = [atom.index for atom in old_ligand_topology.atoms() if atom.element == app.Element.getByAtomicNumber(1)]
    new_hs = [atom.index for atom in new_ligand_topology.atoms() if atom.element == app.Element.getByAtomicNumber(1)]

    def _make_constraint_dict(_system, hs_indices):
        _dict = {}
        for idx in range(_system.getNumConstraints()):
            a1, a2, _len = _system.getConstraintParameters(idx)
            if a1 in hs_indices: _dict[a1] = _len
            if a2 in hs_indices: _dict[a2] = _len
        return _dict

    old_constraint_dict = _make_constraint_dict(old_ligand_system, old_ligand_topology)
    new_constraint_dict = _make_constraint_dict(new_ligand_system, new_ligand_topology)

    to_del = []
    for new_idx, old_idx in new_to_old_map.items():
        if new_idx in new_constraint_dict.keys() and old_idx in old_constraint_dict.keys():
            old_len, new_len = old_constraint_dict[old_idx], old_constraint_dict[new_constraint_dict]
            if old_len != new_len:
                to_del.append(new_idx)
        elif old_idx in old_constraint_dict.keys() and new_idx not in new_constraint_dict.keys():
            to_del.append(new_idx)
        elif old_idx not in old_constraint_dict.keys() and new_idx in new_constraint_dict.keys():
            to_del.append(new_idx)

    for idx in to_del:
        del out_new_to_old_atom_map[idx]

    return out_new_to_old_atom_map


def get_full_atom_map_and_uniques(
        ligand_new_to_old: Dict[int, int],
        old_omm_topology: openmm.app.Topology,
        new_omm_topology: openmm.app.Topology,
        resname: str = 'MOL',
        **unused_kwargs
):
    """given a ligand-only new-to-old atom map (zero-indexed),
    figure out how to expand the map to accommodate mapped `environment` atoms in
    old/new omm_topologies; also return the offset new and old atoms for free."""
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

    # cool, get the unique atoms; I'm assuming here that we have hit the `MOL` residue
    unique_news = [atom for atom in new_indices if atom not in offset_lig_new_to_old_map.keys()]
    unique_olds = [atom for atom in old_indices if atom not in offset_lig_new_to_old_map.values()]

    return new_to_old_atom_map, unique_olds, unique_news


def render_mol_map(mol_old,
                   mol_new,
                   new_to_old_atom_map,
                   align_substructures: bool = True,
                   MolsToGridImage_kwargs: Dict = {'subImgSize': (600, 600)},
                   **unused_kwargs):
    """
    make a mol mapping for two small molecules
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

    img = Draw.MolsToGridImage([mol_old, mol_new], molsPerRow=2, highlightAtomLists=match, **MolsToGridImage_kwargs)

    return img


class RenderSolventOMMObjects(object):
    """A simple handler class to parameterize ligands/complexes into `openmm`
    """

    def __init__(self,
                 ligand_old_sdf: str,  # the `ligand_old.sdf` to query
                 ligand_new_sdf: str,  # the `ligand_new.sdf` to query,
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
                     'padding': 9. * unit.angstroms,
                     'ionicStrength': 0.15 * unit.molar},
                 ):
        # TODO: revise this or make a subclass to handle protein appending...we are omitting this for now
        from openmmforcefields.generators import SystemGenerator

        # make openff.toolkit `molecules` from sdf files
        self._Molecule_old = Molecule_from_sdf(ligand_old_sdf)
        self._Molecule_new = Molecule_from_sdf(ligand_new_sdf)

        # make system gen first since it's used for all phases/endstates
        self._system_generator = SystemGenerator(
            molecules=[self._Molecule_old, self._Molecule_new],
            **SystemGenerator_kwargs)

        # update the `addSolvent` kwargs to update the ff used from `SystemGenerator`
        addSolvent_kwargs.update({'forcefield': self._system_generator.forcefield})

        # reassign name of mol to target
        for _mol in [self._Molecule_old, self._Molecule_new]:
            _ = _mol.name = 'MOL'

        # make `openmm.Topology` for vacuum phase
        self._vacuum_omm_topology_old = self._Molecule_old.to_topology().to_openmm()
        self._vacuum_omm_topology_new = self._Molecule_new.to_topology().to_openmm()

        # get vacuum positions.
        self._vacuum_omm_positions_old = self._Molecule_old._conformers[0]
        self._vacuum_omm_positions_new = self._Molecule_new._conformers[0]

        # make solvent old topology/positions
        self._solvent_omm_topology_old, self._solvent_omm_positions_old = omm_solvate_with_padding(
            omm_topology=self._vacuum_omm_topology_old,
            omm_positions=self._vacuum_omm_positions_old,
            addSolvent_kwargs=addSolvent_kwargs)

        # the solvent new topology, positions are rendered slightly differently; we use the mapping for this
        self._solvent_omm_topology_new = make_new_omm_top(old_omm_topology=self._solvent_omm_topology_old,
                                                          new_resname_omm_topology=self._vacuum_omm_topology_new)

        # generate systems
        self._vacuum_omm_system_old = self._system_generator.create_system(self._vacuum_omm_topology_old)
        self._vacuum_omm_system_new = self._system_generator.create_system(self._vacuum_omm_topology_new)

        self._solvent_omm_system_old = self._system_generator.create_system(self._solvent_omm_topology_old)
        self._solvent_omm_system_new = self._system_generator.create_system(self._solvent_omm_topology_new)