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
import functools
from flax import linen as nn
import jaxopt
import scipy

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

# spatial helper fns
def aperiodic_3D_distance(x: float, y: float, z: float, 
                                    x0: float, y0: float, z0: float) -> float:
    """compute the aperiodic euclidean distance between (x,y,z) and (x0,y0,z0)"""
    r2 = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
    return jnp.sqrt(r2)

def vectorize_aperiodic_3D_distance(X: jnp.array, Y: jnp.array, Z: jnp.array, 
    x0: float, y0: float, z0: float) -> jnp.array:
    """vmap the x, y, z components of `aperiodic_3D_distance` on a 3D grid"""
    out_fn = jax.vmap(
        jax.vmap(
            jax.vmap(aperiodic_3D_distance, in_axes=(0,0,0,None,None,None)),
            in_axes=(0,0,0,None,None,None)),
        in_axes=(0,0,0,None,None,None))
    return out_fn(X, Y, Z, x0, y0, z0)

def cartesian_linspaces_and_retsteps(limits: jnp.array, 
                                     num_gridpoints_per_dim: float) -> Tuple[jnp.array, jnp.array]:
    """given a `limits` (shape [n_dim, 2]) for each dimension and a `num_gridpoints_per_dim` [n_dim],
    generate linspaces and retstep sizes for each dimension"""
    partial_linspaces = functools.partial(jnp.linspace, num = num_gridpoints_per_dim, retstep=True)
    cartesian_linspaces, d_spatial = jax.vmap(partial_linspaces)(limits[:,0], limits[:,1])
    return cartesian_linspaces, d_spatial

def make_cartesian_spatial_grid(limits: jnp.array, 
                                num_gridpoints_per_dim: float, 
                                **unused_kwargs) -> Tuple[jnp.array]:
    """compute a cartesian spatial grid given an `[n_dim, 2]` array of dimensions, an [n_dim] array for `num_gridpoints`.
    returns a tuple. the first is a tuple of (n_dim, [*num_gridpoints]), and the second is a [n_dim] array of spatial 
    gridpoint sizes"""
    cartesian_linspaces, d_spatial = cartesian_linspaces_and_retsteps(limits, num_gridpoints_per_dim)
    cartesian_grids = jnp.meshgrid(*cartesian_linspaces, indexing='ij')
    return cartesian_grids, d_spatial

def reference_posit_r_array(X: jnp.array, Y: jnp.array, Z: jnp.array, 
                             reference_posits: jnp.array) -> jnp.array:
    """given X,Y,Z grids and a [n_points, 3] array of points, compute an [n_points, _X, _Y, _Z] array (where _Q is the number
    of grid points in the Q direction) where each entry along the leading axis is the 
    distance from `n_point` to each grid site"""
    ref_xs, ref_ys, ref_zs = reference_posits[:,0], reference_posits[:,1], reference_posits[:,2]
    out = jax.vmap(vectorize_aperiodic_3D_distance, # vectorize along the reference positions
                   in_axes=(None, None, None, 0,0,0))(X, Y, Z, ref_xs, ref_ys, ref_zs)
    return out

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
    def V_ext_kwarg_generator(particle_idx: int, params: Dict[str, jnp.array], lambda_global: float, **unused_kwargs) -> Dict[str, jnp.array]:
        """generates the vmapped `Dict` for `V_ext`;"""
        unique_idx = map_identical_indices[particle_idx]
        lambda_select = lambda_selector_fn(params, lambda_global, unique_idx)
        _dict = {key: val[particle_idx] for key, val in parameter_dict.items()}
        mod_lam_select = lambda_select_modifier(lambda_select, _dict['uo1'], _dict['un1'], lambda_global)
        _dict['lambda_select'] = mod_lam_select
        return _dict

    return out_params, V_ext_kwarg_generator, V_ext, out_state, lambda_selector_fn

# fingerprinting for unique particles
def compute_atom_centered_fingerprints(mol,
                                       generator,
                                       fpSize,
                                       normalize=True):
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

# loss utilities
def make_loss_utilities(
        solute_positions: jnp.array,
        V_ext_kwarg_generator_fn: Callable[[int, Dict[str, jnp.array], float], Dict[str, jnp.array]],
        V_ext_fn: Callable[[float, Dict[str, jnp.array]], float],  # call params, lambda_global
        grid_limits: jnp.array = jnp.array([[-1., 1.], [-1., 1.], [-1., 1.]]),
        num_gridpoints: int = 64,
        **unused_kwargs) -> Tuple[Callable[float, Callable[[jnp.array, float, Dict, jnp.array], jnp.array]],
jnp.array,
jnp.array]:
    """function to create utilities to place into the loss function;
    returns the following:
    1. `make_vmap_V_ext`: a utility function to create derivatives of vmapped
        (w.r.t. all solute positions and spatial grids) `V_ext` w.r.t. lambda global
    2. `V_ext_R_arr`: an array of shape [num_solute_positions, gridX, gridY, gridZ]
    3. `grid_sizes`: an array of shape [3,] that has the grid sizes in X, Y, Z directions
    """
    # center solute positions
    solute_positions = solute_positions - jnp.mean(solute_positions, axis=0)

    # manage the grid
    (X, Y, Z), (dx, dy, dz) = make_cartesian_spatial_grid(grid_limits,
                                                          num_gridpoints)  # make grids and corresponding spacing
    grid_sizes = jnp.array([dx, dy, dz])
    grid_centers = (grid_limits[:, 0] + grid_limits[:, 1]) / 2.
    R = vectorize_aperiodic_3D_distance(X, Y, Z, *grid_centers)  # make full R matrix [n, n, n]
    V_ext_R_arr = reference_posit_r_array(X, Y, Z,
                                          reference_posits=solute_positions)  # make grid rs [n_solute, _X, _Y, _Z]

    # make V_ext_fn; vmap along solute particles
    def V_ext(r, lambda_global, params, idx):
        V_dict = V_ext_kwarg_generator_fn(idx, params, lambda_global)
        V_dict['lambda_global'] = lambda_global
        return V_ext_fn(r, V_dict)

    # make vmap fn
    def make_vmap_V_ext(num_derivatives):
        """take `num_derivatives` of `V_ext` w.r.t. `lambda_global` (argnum=1)"""
        out_fn = V_ext
        for i in range(num_derivatives):  # take derivatives w.r.t. argnum 1 (lambda_global)
            out_fn = jax.grad(out_fn, argnums=1)
        spatial_arr_fn = jax.vmap(
            jax.vmap(jax.vmap(out_fn, in_axes=(0, None, None, None)), in_axes=(0, None, None, None)),
            in_axes=(0, None, None, None))
        spatial_arr_on_solute_fn = jax.vmap(spatial_arr_fn, in_axes=(0, None, None, 0))
        return spatial_arr_on_solute_fn

    return make_vmap_V_ext, V_ext_R_arr, grid_sizes

def dict_to_array_utilities(
        example_dict: Dict[Any, Any],
        **unused_kwargs) -> Tuple[Callable[Dict, jnp.array], Callable[jnp.array, Dict]]:
    """simple utility functions to convert from a nested pytree to a flat 1D array and back;
    this is useful when using scipy utilities that require numpy arrays instead of arbitrary pytrees"""
    list_init_params, pytree_def = jax.tree_util.tree_flatten(
        example_dict)  # make a list of numpy arrays and a pytree def
    list_shapes = [i.shape for i in list_init_params]  # get the shapes of each array
    flat_list_init_params = [i.flatten() for i in list_init_params]  # flatten each array
    flat_list_lengths = [len(i) for i in flat_list_init_params]  # get the size of each flat array
    flat_init_params = jnp.concatenate(flat_list_init_params)  # concatenate the initial parameters
    list_length_cumsum = jnp.concatenate([jnp.array([0]), jnp.cumsum(jnp.array(flat_list_lengths))])  # get the start_stop indices to gather
    list_indices = [jnp.arange(list_length_cumsum[i], list_length_cumsum[i+1]) for i in range(len(list_length_cumsum)-1)]

    def pytree_to_flat_params(pytree):
        _list, _ = jax.tree_util.tree_flatten(pytree)
        return jnp.concatenate([i.flatten() for i in _list])

    def flat_params_to_pytree(flat_params):
        _list = [jnp.take(flat_params, indices) for indices in list_indices]
        _resh_list = [entry.reshape(shape) for entry, shape in zip(_list, list_shapes)]
        return jax.tree_util.tree_unflatten(pytree_def, _resh_list)

    return pytree_to_flat_params, flat_params_to_pytree

# loss
def make_loss_and_aux(
        initial_parameters_dict: Dict[str, jnp.array],
        make_vmap_V_ext: Callable[float, Callable[[jnp.array, float, Dict[str, jnp.array]], jnp.array]],
        V_ext_R_array: jnp.array,
        grid_sizes: jnp.array,
        n0: float,
        kT: float,
        **unused_kwargs) -> Tuple[Callable]:
    """a function to generate a loss function for parameter optimization;
    functions have to be re-wrapped to make them `scipy.integrate.quad_vec`-compatible
    since an integral over `lambda_global` is needed to compute the loss and its derivative w.r.t. optimizable parameters"""
    num_solute_particles = V_ext_R_array.shape[0]

    # make params flatten/unflatten utils for `quad_vec`
    pytree_to_flat_params, flat_params_to_pytree = dict_to_array_utilities(initial_parameters_dict)

    U = make_vmap_V_ext(0)  # take `V_ext_R_array`, `lambda_global`, `params`, `indices`; do 0 derivatives
    dU_dlam = make_vmap_V_ext(1)  # do 1 derivative
    d2U_dlam2 = make_vmap_V_ext(2)  # do 2 deriviatives

    # NOTE: all derivatives ^ are taken w.r.t. `lambda_global`

    def density(lambda_global: float, parameters: Dict):  # compute the solvent density on grid
        Us = U(V_ext_R_array, lambda_global, parameters, jnp.arange(num_solute_particles)).sum(axis=0)
        return jnp.exp(-Us / kT) * n0  # exponential of (negative) Us gives g0

    def mean_du_dlam(lambda_global: float, parameters: Dict):  # compute the expectation of du/dlambda
        n = density(lambda_global, parameters)
        _mean_du_dlam = dU_dlam(V_ext_R_array, lambda_global, parameters, jnp.arange(num_solute_particles)).sum(
            axis=0) * n / kT
        return jnp.sum(_mean_du_dlam * jnp.prod(grid_sizes))

    def mean_d2u_dlam2(lambda_global: float, parameters: Dict):  # compute the expectation of d2u/dlam2
        n = density(lambda_global, parameters)
        _mean_d2u_dlam2 = d2U_dlam2(V_ext_R_array, lambda_global, parameters, jnp.arange(num_solute_particles)).sum(
            axis=0) * n / kT
        return jnp.sum(_mean_d2u_dlam2 * jnp.prod(grid_sizes))

    def naught_weighted_U(lambda_global,
                          parameters: Dict):  # compute the expectation of U another way (used for validation test)
        n = jax.lax.stop_gradient(density(lambda_global, parameters))
        us = U(V_ext_R_array, lambda_global, parameters, jnp.arange(num_solute_particles)).sum(axis=0) / kT
        return jnp.sum(us * n * jnp.prod(grid_sizes))

    def concat_valgrad_mean_d2u_dlam2_4quadvec(lambda_global: float, flat_params: jnp.array):
        # generate a val/grad concatenated vector of `mean_d2u_dlam2` that is compatible w/ `quad_vec`
        val, grad = jax.value_and_grad(mean_d2u_dlam2, argnums=1)(lambda_global, flat_params_to_pytree(flat_params))
        return jnp.concatenate([jnp.array([val]), pytree_to_flat_params(grad)])

    def valgrad_mean_du_dlam_4quadvec(lambda_global: float, flat_params: jnp.array):
        # generate a val/grad of `mean_du_dlam`
        val, grad = jax.value_and_grad(mean_du_dlam, argnums=1)(lambda_global, flat_params_to_pytree(flat_params))
        return val, grad

    # jit functions used in loss.
    concat_valgrad_mean_d2u_dlam2_4quadvec = jax.jit(concat_valgrad_mean_d2u_dlam2_4quadvec)
    valgrad_mean_du_dlam = jax.jit(valgrad_mean_du_dlam_4quadvec)

    def loss_val_grad(flat_params, **quad_kwargs):  # final loss (val/grad) fn
        dloss_valgrad, dloss_valgrad_err = scipy.integrate.quad_vec(
            concat_valgrad_mean_d2u_dlam2_4quadvec,
            0, 1, args=(flat_params,),
            **quad_kwargs)
        val1, grad1 = valgrad_mean_du_dlam(1., flat_params)
        val0, grad0 = valgrad_mean_du_dlam(0., flat_params)

        out_val = dloss_valgrad[0] - (val1 - val0)
        out_grad = dloss_valgrad[1:] - (grad1 - grad0)
        return out_val, out_grad

    return pytree_to_flat_params, flat_params_to_pytree, density, mean_du_dlam, mean_d2u_dlam2, loss_val_grad, naught_weighted_U