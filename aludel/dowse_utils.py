"""various utilities for dowsing"""
import os
from openmm import unit
import openmm
from typing import List, Any, Set, Callable, Iterable, Tuple, Dict
import jax
from jax import numpy as jnp
import numpy as np
from aludel.utils import maybe_params_as_unitless
from aludel.system_prep_utils import find_res_indices
from openff.toolkit.topology import Molecule

# target rf specifically
#from aludel.rf import SingleTopologyHybridNBFReactionFieldConverter as Converter
DEFAULT_SOFTCORE_PARAMETERS = {
        'softcore_alpha': 0.5,
        'softcore_beta': 0.5,
        'softcore_b': 1,
        'softcore_c': 6,
        'softcore_d': 1,
        'softcore_e': 1,
        'softcore_f': 2
    } # note these are slightly modified from the params in `alude.rf` for stable second derivatives

#spline utilities
# spline stuff
def make_spline_coeffs(x: jnp.array, **unused_kwargs) -> Callable[jnp.array, Tuple[jnp.array]]:
    """given an array of knot points on x axis, return a function that will compute spline coefficients
    given the corresponding values of the y axis knots."""
    num_terms = x.shape[0]
    delta_xs = jnp.diff(x)
    
    indices = jnp.arange(1, num_terms-1, dtype=jnp.int32)
    indices_m1 = (indices - 1)
    indices_p1 = (indices + 1)
    
    indices_list = indices.tolist()
    indices_m1_list = indices_m1.tolist()
    indices_p1_list = indices_p1.tolist()
    
    out_indices = jnp.arange(0, num_terms-1, dtype=jnp.int32)
    out_indices_p1 = out_indices + 1
    
    def spline_coeffs(y: jnp.array, **unused_kwargs) -> Tuple[jnp.array]:
        """returns coefficients for natural cubic splines
        b: coefficients of x, deg 1
        c: coefficients of x, deg 2
        d: coefficients of x, deg 3;
        adapted from : https://medium.com/eatpredlove/natural-cubic-splines-implementation-with-python-edf68feb57aa"""
        delta_ys = jnp.diff(y)
        A = jnp.zeros((num_terms, num_terms))
        b = jnp.zeros((num_terms, 1))
        A = A.at[[0, -1], [0, -1]].set(1.)
        
        A = A.at[(indices_list, indices_m1_list)].set(delta_xs.at[indices_m1].get())
        A = A.at[(indices_list, indices_p1_list)].set(delta_xs.at[indices].get())
        A = A.at[(indices_list, indices_list)].set(2. * (delta_xs.at[indices_m1].get() + delta_xs.at[indices].get()))
    
        b = b.at[(indices_list, 0)].set(3. * (delta_ys.at[indices].get() / delta_xs.at[indices].get() - 
                                              delta_ys.at[indices_m1].get() / delta_xs.at[indices_m1].get()))
    
        # solve for c in Ac = b ; c = A^(-1) * b
        invA = jnp.linalg.inv(A)
        c = jnp.matmul(invA, b).flatten() # column vector to flat
    
        d = (c.at[out_indices_p1].get() - c.at[out_indices].get()) / (3. * delta_xs.at[out_indices].get())
        b = delta_ys.at[out_indices].get()/delta_xs.at[out_indices].get() - (delta_xs.at[out_indices].get()/3.) * (2. * c.at[out_indices].get() + c.at[out_indices_p1].get())
        return b,c,d
    
    return spline_coeffs

def find_indices(arr: jnp.array, value: float) -> Tuple[int, int]:
    """simple utility to pull appropriate knot indices for `exclusive_cubic_spline_fn`"""
    indices = jnp.searchsorted(arr, value)
    out_l, out_r = indices - 1, indices
    out_l = jax.lax.select(out_l < 0, 0 * out_l, out_l) # protect against pulling index < 0
    return out_l, out_r

def exclusive_cubic_spline_fn(x: float, xs: jnp.array, ys: jnp.array, 
                    bs: jnp.array, cs: jnp.array, ds: jnp.array, **unused_kwargs) -> float:
    """given an in x value, find the appropriate knot, and evaluate the cubic function
    defined as S_i(x) = y_i + b_i * (x - x_i) + c_i * (x - x_i)^2 + d_i * (x - x_i)^3 on [x_i, x_{i+1}];
    here, I will presume that x is between xs[0] and xs[-1] inclusively"""
    idx, _ = find_indices(xs, x)
    x_i, y_i = xs[idx], ys[idx]
    return y_i + bs[idx] * (x - x_i) + cs[idx] * (x - x_i)**2 + ds[idx] * (x - x_i)**3


# maker fns
def sc_lj(r, lambda_select, # radius, lambda_select
          os1, os2, ns1, ns2, # old sigma 1, old sigma 2, new sigma 1, new sigma 2
          oe1, oe2, ne1, ne2, # old epsilon 1, old epsilon 2, new epsilon 1, new epsilon 2
          uo1, uo2, un1, un2, # unique old 1, unique old 2, unique new 1, unique new 2
          softcore_alpha, softcore_b, softcore_c, # softcore parameters
          lj_max = 99., **unused_kwargs) -> float: # max lj
    """define a softcore lennard jones potential"""
    # uniques
    uo, un = uo1 + uo2, un1 + un2 # combiner for unique old/new
    unique_old = jax.lax.select(uo >= 1, 1, 0)
    unique_new = jax.lax.select(un >= 1, 1, 0)

    # sigmas
    os = 0.5 * (os1 + os2)
    ns = 0.5 * (ns1 + ns2)

    # epsilons
    oe = jnp.sqrt(oe1 * oe2)
    ne = jnp.sqrt(ne1 * ne2)

    # scaling sigma, epsilon by lambda_select
    res_s = os + lambda_select * (ns - os)
    res_e = oe + lambda_select * (ne - oe)

    # lambda sub for `reff_lj`
    lam_sub = unique_old * lambda_select + unique_new * (1. - lambda_select)
    reff_lj_term1 = softcore_alpha * (lam_sub**softcore_b)
    reff_lj_term2 = (r/res_s)**softcore_c
    reff_lj = res_s * (reff_lj_term1 + reff_lj_term2)**(1./softcore_c)

    # canonical softcore form/protect nans
    lj_x = (res_s / reff_lj)**6.
    lj_e = jnp.array([4. * res_e * lj_x * (lj_x - 1.)])
    lj_e = jnp.nan_to_num(lj_e, nan=jnp.inf)[0]
    lj_e_out = jax.lax.select(lj_e > lj_max, lj_max, lj_e)
    return lj_e_out

def aludel_V_ext_converter(alch_hybrid_particles: List, 
                           alch_nbf: openmm.CustomNonbondedForce, 
                           solvent_nonbonded_params: Dict[str, float],
                           **unused_kwargs):
    """define a template parameter dict compatible with `V_ext`"""
    solvent_nonbonded_params = {key: maybe_params_as_unitless(val)[0] for key, val in solvent_nonbonded_params.items()}
    
    # query the dynamic force's alchemical particles to get the parameters
    parameter_dict = {'os1': [], 'os2': [], 'ns1': [], 'ns2': [], 
                     'oe1': [], 'oe2': [], 'ne1': [], 'ne2': [],
                     'uo1': [], 'uo2': [], 'un1': [], 'un2': []
                     }
    for hybr_idx in alch_hybrid_particles: # gather the particles
        oc, nc, os, ns, oe, ne, uo, un = alch_nbf.getParticleParameters(hybr_idx)
        parameter_dict['os1'].append(os)
        parameter_dict['ns1'].append(ns)
        parameter_dict['oe1'].append(oe)
        parameter_dict['ne1'].append(ne)
        parameter_dict['uo1'].append(uo)
        parameter_dict['un1'].append(un)
    
        # append the solvent params; one for each `alch_hybrid_particle`
        parameter_dict['os2'].append(solvent_nonbonded_params['sigma'])
        parameter_dict['ns2'].append(solvent_nonbonded_params['sigma'])
        parameter_dict['oe2'].append(solvent_nonbonded_params['epsilon'])
        parameter_dict['ne2'].append(solvent_nonbonded_params['epsilon'])
        parameter_dict['uo2'].append(0) # water isn't unique
        parameter_dict['un2'].append(0) # water isn't unique
    
    parameter_dict = {key: jnp.array(val) for key, val in parameter_dict.items()} # make jnp array
    return parameter_dict

def check_identical_particles(identical_particles: np.array, parameter_dict: Dict[str, float]):
    """given an array of int indices, assert that the parameters in the `parameter_dict` are identical
    NOTE: this is an internal test for consistency"""
    if len(identical_particles) > 1:
        ip1 = identical_particles[0]
        template = np.array([val[ip1] for val in parameter_dict.values()])
        for ip in identical_particles[1:]:
            _inner = np.array([val[ip] for val in parameter_dict.values()])
            assert np.allclose(template, _inner), f"{template}, {_inner}"
    else:
        pass

def find_idx(idx: int, list_of_indices: List[jnp.array]) -> int:
    """utility function to pass to `make_lj_V_ext`; 
    returns the unique array index to which a particle index belongs"""
    for _count in range(len(list_of_indices)):
        arr = list_of_indices[_count]
        if idx in arr:
            return _count
    return None

def make_lj_V_ext(
        identical_indices: List[np.array], 
        alch_nbf: openmm.CustomNonbondedForce, 
        solvent_nonbonded_params: Dict[str, float] = {'sigma': [0.3150752406575124 * unit.nanometer],
                                                    'epsilon': [0.635968 * unit.kilojoule_per_mole]},
        softcore_parameters: Dict[str, float] = DEFAULT_SOFTCORE_PARAMETERS,
        num_spline_knots: int=20,
        **unused_kwargs) -> Tuple[jnp.array, Callable[[jnp.array, float], Dict[str, jnp.array]], Callable[float, Dict[str, jnp.array]]]:
    """create a function that generates parameters for the `V_ext` and return init y-axis knots for a cubic spline"""
    num_unique_indices = len(identical_indices)
    alch_hybrid_particles = np.sort(np.concatenate(identical_indices))
    
    # make parameter dict
    parameter_dict = aludel_V_ext_converter(alch_hybrid_particles, 
                                            alch_nbf, 
                                            solvent_nonbonded_params)
    map_identical_indices = jnp.array([find_idx(idx, identical_indices) for idx in alch_hybrid_particles])
    
    # check identical particle parameters
    for identical in identical_indices:
        check_identical_particles(identical, parameter_dict)
        
    # initialize the splines
    x = jnp.linspace(0, 1, num_spline_knots)
    ys_init = jnp.repeat(x[jnp.newaxis, ...], repeats=num_unique_indices, axis=0)
    spline_coeff_fn = make_spline_coeffs(x)

    # make V_ext
    V_ext = lambda r, _dict: sc_lj(r, **_dict, **softcore_parameters)

    def lambda_select_modifier(lambda_select: jnp.array, uo1: int, un1: int, lambda_global: float, **unused_kwargs):
        return jax.lax.select(jnp.isclose(uo1+un1, 0.), lambda_global, lambda_select)
        
    # write function to generate `lambda_select`; xs are partialed out
    def V_ext_kwarg_generator(ys: jnp.array, # [N_unique_indices, num_spline_knots]
                              lambda_global: float, **unused_kwargs) -> Dict[str, jnp.array]:
        """generates the vmapped `Dict` for `V_ext`;"""
        # restrict the y endpoints to 0, 1 and set abs
        ys = ys.at[:,0].set(0.)
        ys = ys.at[:,-1].set(1.) 
        bs, cs, ds = jax.vmap(spline_coeff_fn)(ys)
        lambda_selects = jnp.abs(jax.vmap(exclusive_cubic_spline_fn, 
                                  in_axes=(None,None, 0, 0, 0, 0))(lambda_global, x, ys, bs, cs, ds)) # positive
        _dict = {key: val for key, val in parameter_dict.items()}
        _lam_selects = jnp.array([lambda_selects[q] for q in map_identical_indices])
        mod_lam_selects = jax.vmap(lambda_select_modifier, in_axes=(0,0,0,None))(_lam_selects, _dict['uo1'], _dict['un1'], lambda_global)
        _dict['lambda_select'] = mod_lam_selects
        return _dict

    return ys_init, V_ext_kwarg_generator, V_ext

# fingerprinting for unique particles
def compute_atom_centered_fingerprints(mol,
                                       generator,
                                       fpSize,
                                       normalize = True):
    """
    compute an atom-centric fingerprint of a molecule. You need `rdkit`

    Arguments:
    mol : rdkit.Chem.Mol
    generator : return of `rdFingerprintGenerator.GetCountFingerPrint
    fpSize : size of fingerprint
    normalize : reduce so that all output vals are <= 1.

    Return:
    fingerprints : np.array(mol.GetNumAtoms(), fpSize, dtype=np.float64)

    TODO : fix thee typing (do we need to import `rdkit` here?)

    Example:
    >>> import rdkit
    >>> import numpy as np
    >>> #print(rdkit.__version__)
    >>> from rdkit import Chem
    >>> from rdkit.Chem import RDKFingerprint
    >>> from rdkit.Chem import rdFingerprintGenerator
    >>> mol = Chem.SDMolSupplier('mol.sdf', removeHs=False)[0] # this assumes you want the 0th mol from an sdf called `mol.sdf`
    >>> fpSize = 32
    >>> generator = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=5, maxPath=5, fpSize=fpSize)
    >>> X = compute_atom_centered_fingerprints(mol, generator, fpSize, normalize=True)
    """
    n_atoms = mol.GetNumAtoms()
    fingerprints = np.zeros((n_atoms, fpSize), dtype=int)

    for i in range(mol.GetNumAtoms()):
        fingerprint = generator.GetCountFingerprint(mol, fromAtoms=[i])
        for (key, val) in fingerprint.GetNonzeroElements().items():
            fingerprints[i, key] = val

    fp = np.array(fingerprints, dtype=np.float64)
    if normalize:
        _max = np.max(fp)
        fp = fp / _max

    return fp

def make_identical_particles_list(
    old_omm_topology: openmm.app.topology, 
    new_omm_topology: openmm.app.topology,
    old_to_hybrid_map: Dict[str, int],
    new_to_hybrid_map: Dict[str, int],
    old_mol: Molecule,
    new_mol: Molecule,
    unique_olds: List[int],
    unique_news: List[int],
    hybrid_positions: unit.Quantity,
    truncate_threshold: float,
    resname: str='MOL',
    **unused_kwargs) -> List[jnp.array]:
    """create a list of jnp.array particle indices (of the appropriate alchemical system) 
    to identify potentially identical unique old/new particles; also return appropriate positions"""
    from rdkit.Chem import RDKFingerprint
    from rdkit.Chem import rdFingerprintGenerator
    from aludel.system_prep_utils import compute_distance_matrix

    hybrid_to_old_map = {val: key for key, val in old_to_hybrid_map.items()}
    hybrid_to_new_map = {val: key for key, val in new_to_hybrid_map.items()}

    old_mol_res_indices = find_res_indices(old_omm_topology, resname)
    new_mol_res_indices = find_res_indices(new_omm_topology, resname)
    
    fpSize = 16
    generator = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=5, maxPath=5, fpSize=fpSize)
    old_mol_fp = compute_atom_centered_fingerprints(old_mol.to_rdkit(), generator, fpSize, normalize=False)
    new_mol_fp = compute_atom_centered_fingerprints(new_mol.to_rdkit(), generator, fpSize, normalize=False)

    old_idx_to_fp = {idx: old_mol_fp[_counter] for _counter, idx in enumerate(old_mol_res_indices)}
    new_idx_to_fp = {idx: new_mol_fp[_counter] for _counter, idx in enumerate(new_mol_res_indices)}

    old_hybr_indices = [old_to_hybrid_map[i] for i in old_mol_res_indices]
    new_hybr_indices = [new_to_hybrid_map[i] for i in new_mol_res_indices]

    all_hybr_indices = np.sort(np.unique(np.concatenate((np.array(old_hybr_indices), np.array(new_hybr_indices)))))
    
    identical_unique_particles = []
    rendered = []
    for hybr_idx in all_hybr_indices:
        old_idx = hybrid_to_old_map.get(hybr_idx, -1)
        new_idx = hybrid_to_new_map.get(hybr_idx, -1)
        fp_mapper = old_idx_to_fp if old_idx >= 0 else new_idx_to_fp
        full_fp = old_mol_fp if old_idx >=0 else new_mol_fp
        mapped = True if old_idx >=0 and new_idx >= 0 else False
        back_map = old_to_hybrid_map if old_idx >= 0 else new_to_hybrid_map
        if mapped: # do nothing
            identical_unique_particles.append([hybr_idx])
            rendered += [hybr_idx]
        else: 
            base_idx = old_idx if old_idx >= 0 else new_idx
            fp = fp_mapper[base_idx]
            match_dict = {idx: np.array_equal(fp, _fp) for idx, _fp in fp_mapper.items()}
            quad = []
            for key, val in match_dict.items():
                if val:
                    quad.append(back_map[key])
            to_list = [q for q in quad if q not in rendered]
            rendered += to_list
            if len(to_list) > 0:
                identical_unique_particles.append(to_list)
    
    # now attempt to truncate...
    rendered_arr = np.array(rendered)
    hybrid_p = hybrid_positions.value_in_unit_system(unit.md_unit_system)[rendered_arr,:]
    hybrid_dist_arr = compute_distance_matrix(hybrid_p, hybrid_p)
    uniques_as_hybrid = [old_to_hybrid_map[q] for q in unique_olds] + [new_to_hybrid_map[q] for q in unique_news]
    indices_to_pull = uniques_as_hybrid # np.array([rendered.index(q) for q in uniques_as_hybrid])
    interest_distances = hybrid_dist_arr[indices_to_pull, :]
    positives = interest_distances <= truncate_threshold
    where_thresh = np.any(positives, axis=0)
    to_keep = [rendered_arr[i] for i in np.where(where_thresh)[0]] # sep the tuple
    assert all([u in to_keep for u in uniques_as_hybrid]) # assert all the uniques are still retained

    # filter identical unique particles
    out_unique_particles = []
    for _list in identical_unique_particles:
        mods = [i for i in _list if i in to_keep]
        if len(mods) > 0:
            out_unique_particles.append(mods)

    out_posits = hybrid_positions[to_keep, :]
    return out_unique_particles, out_posits.value_in_unit_system(unit.md_unit_system)