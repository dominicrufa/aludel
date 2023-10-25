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
from flax import linen as nn
import jaxopt

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
def pw_lin_to_quad_to_const(x: float, x_sw: float, y_max: float, **unused_kwargs):
    """define a piecewise function of x s.t.
    define a quadratic term w/ a y_max; 
    the y_max defines an x_sw2 = 2*y_max - x_sw;
    quadratic is defined by y = a * (x - x_sw2)^2 + y_max w/ a = 1. / (2. * (x_sw1 - x_sw2))
    1. less than x_sw, return x;
    2. between x_sw and x_sw2, return quadratic;
    3. greater than x_sw2, return y_max
    """
    x_sw2 = 2 * y_max - x_sw # define x_sw2
    a = 1. / (2. * (x_sw - x_sw2)) # eval a for quadratic
    quad = lambda _x: a * (_x - x_sw2)**2 + y_max # define quad fn
    lin_to_quad = lambda _x: jax.lax.select(_x < x_sw, _x, quad(_x)) # define linear_to_quadratic w/ lower bound

    out = jax.lax.select(x > x_sw2,
                        jnp.array([y_max]),
                        jnp.array([lin_to_quad(x)]))[0]
    return out

def sc_lj(r, lambda_select, # radius, lambda_select
          os1, os2, ns1, ns2, # old sigma 1, old sigma 2, new sigma 1, new sigma 2
          oe1, oe2, ne1, ne2, # old epsilon 1, old epsilon 2, new epsilon 1, new epsilon 2
          uo1, uo2, un1, un2, # unique old 1, unique old 2, unique new 1, unique new 2
          softcore_alpha: float=DEFAULT_SOFTCORE_PARAMETERS['softcore_alpha'], 
          softcore_b: float=DEFAULT_SOFTCORE_PARAMETERS['softcore_b'], 
          softcore_c: float=DEFAULT_SOFTCORE_PARAMETERS['softcore_c'], # softcore parameters
          lj_switch: float=79., 
          lj_max: float=99., 
          **unused_kwargs) -> float: # max lj
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
    #reff_lj = jax.lax.select(reff_lj <- 1e-6, 1e-6, reff_lj) # protect small reff_lj

    # canonical softcore form/protect nans
    lj_x = (res_s / reff_lj)**6
    lj_e = 4. * res_e * lj_x * (lj_x - 1.)
    lj_e = jnp.nan_to_num(lj_e, nan=jnp.inf)
    lj_e = pw_lin_to_quad_to_const(lj_e, lj_switch, lj_max) # add switching so its second order differentiable
    return lj_e

def sc_v2(r, lambda_global, lambda_select, # radius, lambda_select
          os1, os2, ns1, ns2, # old sigma 1, old sigma 2, new sigma 1, new sigma 2
          oe1, oe2, ne1, ne2, # old epsilon 1, old epsilon 2, new epsilon 1, new epsilon 2
          uo1, uo2, un1, un2, # unique old 1, unique old 2, unique new 1, unique new 2
          softcore_alpha: float=DEFAULT_SOFTCORE_PARAMETERS['softcore_alpha'], 
          softcore_b: float=DEFAULT_SOFTCORE_PARAMETERS['softcore_b'], 
          softcore_c: float=DEFAULT_SOFTCORE_PARAMETERS['softcore_c'], # softcore parameters
          lj_switch: float=79., 
          lj_max: float=99., 
          **unused_kwargs) -> float: # max lj
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

    # scaling sigma, epsilon by lambda_global
    res_s = os + lambda_global * (ns - os)
    res_e = oe + lambda_global * (ne - oe)

    # lambda sub for `reff_lj`
    lam_sub = unique_old * lambda_select * lambda_global + unique_new * lambda_select * (1. - lambda_global)
    reff_lj = r + lam_sub

    # canonical softcore form/protect nans
    lj_x = (res_s / reff_lj)**6
    lj_e = 4. * res_e * lj_x * (lj_x - 1.)
    lj_e = jnp.nan_to_num(lj_e, nan=jnp.inf)
    lj_e = pw_lin_to_quad_to_const(lj_e, lj_switch, lj_max) # add switching so its second order differentiable
    return lj_e

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

# NN stuff
class MLP(nn.Module):
    num_hidden_layers: int=1
    dense_features: int=32
    output_dimension: int=1
    
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.dense_features)(x)
            x = nn.swish(x)
        return nn.Dense(self.output_dimension)(x)

class LambdaSelector(nn.Module):
    num_hidden_layers: int=1
    dense_features: int=32
    
    @nn.compact
    def __call__(self, _lambda_global, unique_idx, max_idx):
        x = jnp.array([_lambda_global, unique_idx / (max_idx+1)]) # protect against nan
        spec_val = MLP(num_hidden_layers=self.num_hidden_layers, 
                       dense_features=self.dense_features, 
                       output_dimension=1)(x)
        return spec_val

def make_lambda_selector_fn(max_idx: int,
                            pre_linearize: bool=True, 
                            num_hidden_layers: int=1, 
                            dense_features: int=32, 
                            **unused_kwargs) -> Callable[[float, int, int], float]:
    """initialize and (maybe) pretrain an instance of `LambdaSelector`"""
    model = LambdaSelector(num_hidden_layers, dense_features)
    inits = (.5, 1, max_idx) # lam_global, idx, max_idx
    untrained_params = model.init(jax.random.PRNGKey(42), *inits)
    try_out = model.apply(untrained_params, *inits) # run it for good measure
    
    # define the function
    def lambda_selector_fn(params, lambda_global, idx):
        return model.apply(params, lambda_global, idx, max_idx).sum()**2

    # linearize the function
    if pre_linearize:
        def _loss_fn(params, lambda_global, idx):
            learned_val = lambda_selector_fn(params, lambda_global, idx)
            target_val = DEFAULT_SOFTCORE_PARAMETERS['softcore_alpha']
            return (learned_val - target_val)**2

        loss_over_idx = jax.vmap(_loss_fn, in_axes=(None, None, 0))
        loss_over_lam_glob = jax.vmap(loss_over_idx, in_axes=(None, 0, None))
        loss_fn = lambda params: loss_over_lam_glob(params, jnp.linspace(0, 1, 100), jnp.arange(0, max_idx+1)).sum()
        
        # optimizer
        solver = jaxopt.LBFGS(loss_fn)
        res = solver.run(untrained_params)
        out_params, out_state = res.params, res.state
    else:
        out_params, out_state = untrained_params, None
    return lambda_selector_fn, out_params, untrained_params, out_state

# now the meat
def make_lj_V_ext(
    identical_indices: List[np.array], 
    alch_nbf: openmm.CustomNonbondedForce, 
    solvent_nonbonded_params: Dict[str, float] = {'sigma': [0.3150752406575124 * unit.nanometer],
                                                'epsilon': [0.635968 * unit.kilojoule_per_mole]},
    softcore_parameters: Dict[str, float] = DEFAULT_SOFTCORE_PARAMETERS,
    pre_linearize: bool=True, 
    num_hidden_layers: int=1, 
    dense_features: int=32, 
    **unused_kwargs) -> Tuple[jnp.array, Callable[[jnp.array, float], Dict[str, jnp.array]], Callable[float, Dict[str, jnp.array]]]:
    """create a function that generates parameters for the `V_ext` and return init y-axis knots for a cubic spline"""
    num_unique_indices = len(identical_indices) # get the number of unique indices
    alch_hybrid_particles = np.sort(np.concatenate(identical_indices)) # sort all the alchemical hybrid particles
    
    # make parameter dict
    parameter_dict = aludel_V_ext_converter(alch_hybrid_particles, 
                                            alch_nbf, 
                                            solvent_nonbonded_params)
    # determine the mapping indices of the `alch_hybrid_particles`
    map_identical_indices = jnp.array([find_idx(idx, identical_indices) for idx in alch_hybrid_particles])
    
    # check identical particle parameters as a consistency test
    for identical in identical_indices:
        check_identical_particles(identical, parameter_dict)

    # make V_ext
    V_ext = lambda r, _dict: sc_v2(r, **_dict, **softcore_parameters)

    # make a selection function that replaces learned `lambda_select` with `lambda_global` if the solute is not unique 
    def lambda_select_modifier(lambda_select: jnp.array, uo1: int, un1: int, lambda_global: float, **unused_kwargs):
        return jax.lax.select(jnp.isclose(uo1+un1, 0.), lambda_global, lambda_select)

    # create lambda selector_fn
    lambda_selector_fn, out_params, untrained_params, out_state = make_lambda_selector_fn(
        max_idx = num_unique_indices-1,
        pre_linearize = pre_linearize, 
        num_hidden_layers = num_hidden_layers, 
        dense_features = dense_features)
        
    # write function to generate `lambda_select`; xs are partialed out
    def V_ext_kwarg_generator(params: Dict[str, jnp.array], lambda_global: float, **unused_kwargs) -> Dict[str, jnp.array]:
        """generates the vmapped `Dict` for `V_ext`;"""
        lambda_selects = jax.vmap(lambda_selector_fn, in_axes=(None, None, 0))(params, lambda_global, jnp.arange(num_unique_indices))
        _dict = {key: val for key, val in parameter_dict.items()}
        _lam_selects = jnp.array([lambda_selects[q] for q in map_identical_indices])
        mod_lam_selects = jax.vmap(lambda_select_modifier, in_axes=(0,0,0,None))(_lam_selects, _dict['uo1'], _dict['un1'], lambda_global)
        _dict['lambda_select'] = mod_lam_selects
        _dict['lambda_global'] = jnp.repeat(lambda_global, len(mod_lam_selects))
        return _dict

    return out_params, V_ext_kwarg_generator, V_ext, out_state, lambda_selector_fn

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