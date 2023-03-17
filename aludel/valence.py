"""valence conversion utilities"""
import openmm
from openmm import app, unit
import numpy as np
import copy
from typing import Any, Tuple, Dict, Iterable, Callable


class SingleTopologyHybridValenceConverter(object):
    """class to convert old/new valence force to a hybrid valence force"""
    DYNAMIC_VALENCE_TEMPLATE = [
        "U_old + lambda_global*(U_new - U_old);",
        "U_old = U_valence_old*unique_retainer;",
        "U_new = U_valence_new*unique_retainer;",
        "unique_retainer = select(1-mapped, retain_uniques_delimiter, 1.);",
        "retain_uniques_delimiter = select(1-retain_uniques, unique_selector, 1);",
        "unique_selector = select(1-unique_old, lambda_global, 1-lambda_global);"]
    STATIC_VALENCE_TEMPLATE = ['U_valence_;']
    GLOBAL_PARAMETERS = {'lambda_global': 0, 'retain_uniques': 1.}
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
            'per_term_params': ["length_{end}", "k_{end}"],
            'U_valence_template': "U_valence_{end} = 0.5*k_{end}*(r-length_{end})^2;"},
        'HarmonicAngleForce': {
            'addTerm': 'addAngle',
            'setTerm': 'setAngleParameters',
            'query_num_terms': 'getNumAngles',
            'query_params': 'getAngleParameters',
            'custom_force': openmm.CustomAngleForce,
            'num_particles': 3,
            'add_per_term_param': 'addPerAngleParameter',
            'num_params': 2,
            'per_term_params': ["angle_{end}", "k_{end}"],
            "U_valence_template": "U_valence_{end} = 0.5*k_{end}*(theta-angle_{end})^2;"},
        'PeriodicTorsionForce': {
            'addTerm': 'addTorsion',
            'setTerm': 'setTorsionParameters',
            'query_num_terms': 'getNumTorsions',
            'query_params': 'getTorsionParameters',
            'custom_force': openmm.CustomTorsionForce,
            'num_particles': 4,
            'add_per_term_param': 'addPerTorsionParameter',
            'num_params': 3,
            'per_term_params': ["periodicity_{end}", "phase_{end}", "k_{end}"],
            "U_valence_template": "U_valence_{end} = k_{end}*(1 + cos(periodicity_{end}*theta - phase_{end}));"}
    }
    AUX_PER_TERM_PARAMETERS = ['unique_old', 'unique_new', 'mapped']

    def __init__(self: Any,
                 old_force: openmm.Force,
                 new_force: openmm.Force,
                 old_to_hybrid_map: Dict[int, int],
                 new_to_hybrid_map: Dict[int, int],
                 num_hybrid_particles: int,
                 unique_old_atoms: Iterable[int],
                 unique_new_atoms: Iterable[int],
                 **kwargs):
        # assert that the forces match
        assert old_force.__class__.__name__ == new_force.__class__.__name__
        self._force_name = old_force.__class__.__name__

        self._old_force = old_force
        self._new_force = new_force
        self._old_to_hybrid_map = old_to_hybrid_map
        self._new_to_hybrid_map = new_to_hybrid_map
        self._num_hybrid_particles = num_hybrid_particles
        self._unique_old_atoms = unique_old_atoms
        self._unique_new_atoms = unique_new_atoms
        self._force_utils = self.VALENCE_FORCE_UTILS[self._force_name]

        self._hybrid_to_old_map = {val: key for key, val in self._old_to_hybrid_map.items()}
        self._hybrid_to_new_map = {val: key for key, val in self._new_to_hybrid_map.items()}
        self._hybrid_forces = self._make_hybrid_forces(**kwargs)

    @property
    def hybrid_forces(self):
        return copy.deepcopy(self._hybrid_forces)

    def _make_custom_dynamic_expression(self, **unused_kwargs) -> str:
        """make the custom expression of the custom valence force"""
        base_expr = ' '.join(self.DYNAMIC_VALENCE_TEMPLATE)
        valence_template = self._force_utils['U_valence_template']
        aux_expr_template = [valence_template.format(**{'end': _endstate})
                             for _endstate in ['old', 'new']]
        aux_expr = ' '.join(aux_expr_template)
        return base_expr + aux_expr

    def _make_custom_static_expression(self, **unused_kwargs) -> str:
        """make the custom expression for the static valence force"""
        base_expr = ' '.join(self.STATIC_VALENCE_TEMPLATE)
        valence_template = self._force_utils['U_valence_template']
        aux_expr = valence_template.format(**{'end': ''})
        return base_expr + aux_expr

    def _format_dynamic_per_term_params(self, **unused_kwargs) -> Iterable[str]:
        """return a list of per{Term}Parameters"""
        per_term_param_template = self._force_utils['per_term_params']
        per_term_parameters = [
            [_term.format(end=_endstate) for _term in per_term_param_template]
            for _endstate in ['old', 'new']]
        per_term_parameters = [item for sublist in per_term_parameters
                               for item in sublist] + self.AUX_PER_TERM_PARAMETERS
        return per_term_parameters

    def _format_static_per_term_params(self, **unused_kwargs) -> Iterable[str]:
        """return a list of per{Term}Parameters for the static force"""
        per_term_param_template = self._force_utils['per_term_params']
        per_term_parameters = [_term.format(end='') for _term in per_term_param_template]
        return per_term_parameters

    def _make_hybrid_forces(self, **unused_kwargs):
        """main method that will create the hybrid force"""
        from aludel.utils import maybe_params_as_unitless
        num_particles = self._force_utils['num_particles']

        # make the custom valence forces and add per term params
        custom_dynamic_expr = self._make_custom_dynamic_expression(**unused_kwargs)
        custom_static_expr = self._make_custom_static_expression(**unused_kwargs)

        U_out_dynamic = self._force_utils['custom_force'](custom_dynamic_expr)
        U_out_static = self._force_utils['custom_force'](custom_static_expr)
        is_periodic = self._old_force.usesPeriodicBoundaryConditions()

        _ = U_out_dynamic.setUsesPeriodicBoundaryConditions(is_periodic)
        _ = U_out_static.setUsesPeriodicBoundaryConditions(is_periodic)

        for gp_name, gp_val in self.GLOBAL_PARAMETERS.items():  # GlobalParameters
            _ = U_out_dynamic.addGlobalParameter(gp_name, gp_val)

        add_per_term_param_fn_dynamic = getattr(U_out_dynamic,
                                                self._force_utils['add_per_term_param'])
        add_per_term_param_fn_static = getattr(U_out_static,
                                               self._force_utils['add_per_term_param'])

        # defining and adding per-term parameters
        per_term_parameters_dynamic = self._format_dynamic_per_term_params(**unused_kwargs)
        per_term_parameters_static = self._format_static_per_term_params(**unused_kwargs)
        for _term in per_term_parameters_dynamic:  # perTermParameters
            _ = add_per_term_param_fn_dynamic(_term)
        for _term in per_term_parameters_static:
            _ = add_per_term_param_fn_static(_term)

        # get the setTerm/addTerm/query methods for dynamic force
        term_adder_fn_dynamic = getattr(U_out_dynamic, self._force_utils['addTerm'])
        term_setter_fn_dynamic = getattr(U_out_dynamic, self._force_utils['setTerm'])
        query_U_out_by_idx_fn_dynamic = getattr(U_out_dynamic, self._force_utils['query_params'])

        # get the setTerm/addTerm/query methods for static force
        term_adder_fn_static = getattr(U_out_static, self._force_utils['addTerm'])

        _num_parameters = self._force_utils['num_params']

        # make a copy of `U_out_dynamic` which will actually be returned;
        # `U_out_dynamic` is actually just a placeholder
        U_out_dynamic2 = copy.deepcopy(U_out_dynamic)
        term_adder_fn_dynamic2 = getattr(U_out_dynamic2, self._force_utils['addTerm'])

        hybr_idx_to_term_dict = {}  # record dict of added terms
        for force in [self._old_force, self._new_force]:  # iterate over old/new force
            query_term_by_idx_fn = getattr(force, self._force_utils['query_params'])
            num_terms = getattr(force, self._force_utils['query_num_terms'])()
            uniques = self._unique_old_atoms if force == self._old_force else self._unique_new_atoms
            to_hybrid_map = self._old_to_hybrid_map if force == self._old_force else self._new_to_hybrid_map
            for idx in range(num_terms):
                all_params = query_term_by_idx_fn(idx)  # full param spec
                orig_indices = all_params[:num_particles]  # get the original indices
                hybr_indices = [to_hybrid_map[_q] for _q in orig_indices]  # map to hybrid indices
                unitless_term_params = maybe_params_as_unitless(all_params[num_particles:])  # make the terms unitless
                hybr_ind_str = '.'.join([str(_q) for _q in hybr_indices])  # make hybrid indices strings; order matters
                # check if contains uniques first
                if len(set(orig_indices).intersection(uniques)) > 0:
                    # unique new/old particles included
                    aux_delim = [1, 0, 0] if force == self._old_force else [0, 1, 0]
                    # for uniques, always retain endstates; add to dynamic custom force
                    _term_idx = term_adder_fn_dynamic(*hybr_indices,
                                                      unitless_term_params + unitless_term_params + aux_delim)
                else:  # no unique particles in this.
                    if force == self._old_force:  # iterate over this first,
                        # make it old; adjust this afterward if it is
                        # encountered in `force == new_force`
                        _term_idx = term_adder_fn_dynamic(*hybr_indices,
                                                          unitless_term_params + [_q * 0. for _q in
                                                                                  unitless_term_params] + [0, 0,
                                                                                                           1])
                        try:  # try to append it to the recorder dict
                            hybr_idx_to_term_dict[hybr_ind_str].append(_term_idx)
                        except Exception as e:  # if it is not in the keys, make a new one
                            hybr_idx_to_term_dict[hybr_ind_str] = [_term_idx]
                    else:  # force == new_force; iterating over new force now
                        # try to query the hybrid indices in forward/reverse direction
                        rev_hybr_ind_str = '.'.join([str(_q) for _q in hybr_indices[::-1]])
                        try:  # try to query the recorder dict
                            try:  # query as the str exists
                                match_term_indices = hybr_idx_to_term_dict[hybr_ind_str]
                                _fwd_key, _bkwd_key = (True, False)
                            except:  # query backward str
                                match_term_indices = hybr_idx_to_term_dict[rev_hybr_ind_str]
                                _fwd_key, _bkwd_key = (False, True)
                        except Exception as e:  # there are no match terms
                            match_term_indices = []
                        if len(match_term_indices) == 0:
                            # this is a mapped term that is a "false" unique new term
                            # in that it is a valence term without uniques that appears in
                            # the new system and not the old system
                            _term_idx = term_adder_fn_dynamic(*hybr_indices,
                                                              [_q * 0. for _q in
                                                               unitless_term_params] + unitless_term_params + [0,
                                                                                                               0,
                                                                                                               1])
                        else:  # there is at least 1 idx match; now match parameters
                            param_match = False
                            for _counter, match_term_idx in enumerate(match_term_indices):  # iterate over matches
                                match_params = query_U_out_by_idx_fn_dynamic(match_term_idx)
                                non_delimiter_match_params = list(match_params[-1][:-(3 + _num_parameters)])
                                if np.allclose(list(unitless_term_params), list(non_delimiter_match_params)):
                                    # these terms match, and we can make them static...
                                    param_match = True
                                    break
                            if param_match:  # there is a term with _exact_ parameters
                                # add to dynamic force as a placeholder. and remove the index so it isn't double counted
                                _ = term_setter_fn_dynamic(match_term_idx, *hybr_indices,
                                                           list(non_delimiter_match_params) + list(
                                                               non_delimiter_match_params) + [0,
                                                                                              0,
                                                                                              -1])
                                if _fwd_key:
                                    del hybr_idx_to_term_dict[hybr_ind_str][_counter]
                                elif _bkwd_key:
                                    del hybr_idx_to_term_dict[rev_hybr_ind_str][_counter]
                                else:
                                    raise Exception(f"There is a mishandling of the counter for matched terms...")
                            else:  # there is no term with _exact_ parametes; add to new
                                _ = term_adder_fn_dynamic(*hybr_indices,
                                                          [_q * 0. for _q in
                                                           unitless_term_params] + unitless_term_params + [0, 0, 1])

        # iterate back over `U_out_dynamic`; copy all consistent terms into static
        # copy all inconsistent terms into `U_out_dynamic2`
        num_dynamic_terms = getattr(U_out_dynamic, self._force_utils['query_num_terms'])()
        for idx in range(num_dynamic_terms):
            _params = query_U_out_by_idx_fn_dynamic(idx)  # query the dynamic function by idx
            hybrid_indices = _params[:-1]
            if _params[-1][-1] == -1:  # we previously set `mapped` indicator to `-1` if the
                # term matched the old and new (cannot be unique)
                term_params = _params[-1]
                _ = term_adder_fn_static(*hybrid_indices, term_params[:_num_parameters])
            else:
                _ = term_adder_fn_dynamic2(*_params)

        return [U_out_static, U_out_dynamic2]
