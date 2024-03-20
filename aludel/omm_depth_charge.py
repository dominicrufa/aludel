"""openmm routines for depth charge; requires openmm and openmm"""
import openmm
from openmm import unit
import openmmtools
import jax
from jax import numpy as jnp
import numpy as np
import copy
import typing
import tqdm

# aludel imports
from aludel.atm import SCRFSingleTopologyHybridSystemFactory

# constants
DEFAULT_BAROSTAT = openmm.MonteCarloBarostat(1.01325, 300., 25)

def make_platform(platform_name: str = 'CUDA', **unused_kwargs) -> openmm.Platform:
    """get an appropriate platform"""
    platform = openmm.Platform.getPlatformByName(platform_name)
    platform_name = platform.getName()
    if platform_name in ['CUDA']:
        platform.setPropertyDefaultValue('Precision', 'mixed')
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
    print(f"using {platform_name} platform")
    return platform 

def compute_radial_distances_and_velocities(
    positions: jax.Array, 
    velocities: jax.Array,
    indices_to_query: jax.Array, 
    fixed_site_idx: int, 
    epsilon: float = 1e-8,
    **unused_kwargs) -> jax.Array:
    """compute the radial positions (and velocities) of given particles w.r.t. a fixed site (typically used w/ `enforcePeriodicBox=True`);
    the typical use case is where `fixed_site_idx` is centered at the origin."""
    # positions first
    fixed_site_position = positions[fixed_site_idx]
    positions_to_query = positions[indices_to_query]
    vector_distances = positions_to_query - fixed_site_position
    r2s = jax.vmap(jnp.dot)(vector_distances, vector_distances)
    rs = jnp.sqrt(r2s)

    # velocities second
    radial_unit_vectors = vector_distances / (rs[..., jnp.newaxis] + epsilon)
    radial_velocities = jax.vmap(jnp.dot)(velocities[indices_to_query], radial_unit_vectors)
    return rs, radial_velocities

def make_solvated_particle(lj_charges: typing.List = [0., 0.], 
                          lj_sigmas: typing.List = [0.3150752406575124, 0.3150752406575124],
                          lj_epsilons: typing.List = [0., 0.635968], 
                          barostat: typing.Union[None, openmm.MonteCarloBarostat] = copy.deepcopy(DEFAULT_BAROSTAT),
                          hmass: unit.daltons = 3. * unit.daltons,
                          wb_kwargs: typing.Dict[str, typing.Any] = {'box_edge': 3.5 * unit.nanometer},
                          **unused_kwargs) -> openmm.System:
    """make a simple `openmm.System` with a lj particle centered at the origin (infinite mass);
    NOTE: if the old(new) charges and epsilons are close to zero, then the lj particle will be unique, core otherwise
    """
    
    wb = openmmtools.testsystems.WaterBox(**wb_kwargs)
    old_system = wb.system
    bvs = wb.system.getDefaultPeriodicBoxVectors()
    bvs = np.array([i.value_in_unit_system(unit.md_unit_system) for i in bvs])
    new_position = 0.5 * np.array([bvs[0,0], bvs[1,1], bvs[2,2]]) # place at center of periodic box
    

    # reset particle masses
    oxy_mass = 16 * unit.daltons - 2*(hmass) # HMass Repartition.
    for i in range(old_system.getNumParticles()):
        if old_system.getParticleMass(i) < hmass: # is hydrogen
            old_system.setParticleMass(i, hmass)
        else:
            old_system.setParticleMass(i, oxy_mass)
            
    if barostat is not None: # whether to add a barostat
        old_system.addForce(barostat)
    new_system = copy.deepcopy(old_system)

    # make a new topology for rendering purposes
    new_topology = copy.deepcopy(wb.topology)
    my_chain = list(wb.topology.chains())[0]
    new_residue = new_topology.addResidue('lj', my_chain)
    new_element = list(wb.topology.atoms())[0].element # will be oxy
    new_atom = new_topology.addAtom('_lj', new_element, new_residue)

    # mod the new system to add a particle
    unique_new = np.isclose(lj_charges[0], 0.) and np.isclose(lj_epsilons[0], 0.)
    unique_old = np.isclose(lj_charges[1], 0.) and np.isclose(lj_epsilons[1], 0.)
    assert not unique_old, f"doesn't currently support unique old because that would make it redundant"
    core = not unique_new # only this if parameters simply change from nonzero to nonzero
    if unique_new:
        new_atom_idx = new_system.addParticle(0)
        new_nbf = [f for f in new_system.getForces() if f.__class__.__name__ == 'NonbondedForce'][0]
        new_nbf.addParticle(lj_charges[1], lj_sigmas[1], lj_epsilons[1])
        old_to_new_atom_map = {i: i for i in range(old_system.getNumParticles())}
        unique_old_atoms = []
        unique_new_atoms = [new_atom_idx] # there is only 1 particle being added.
        old_positions = wb.positions
        new_positions = np.concatenate((old_positions.value_in_unit_system(unit.md_unit_system), new_position[np.newaxis,...])) * unit.nanometer
    else: # core
        _ = old_system.addParticle(0)
        _ = new_system.addParticle(0)
        old_nbf = [f for f in old_system.getForces() if f.__class__.__name__ == 'NonbondedForce'][0]
        new_nbf = [f for f in new_system.getForces() if f.__class__.__name__ == 'NonbondedForce'][0]
        old_to_new_atom_map = {i: i for i in range(old_system.getNumParticles())}
        unique_old_atoms, unique_new_atoms = [], []
        _ = old_nbf.addParticle(lj_charges[0], lj_sigmas[0], lj_epsilons[0])
        _ = new_nbf.addParticle(lj_charges[1], lj_sigmas[1], lj_epsilons[1])
        positions = wb.positions.value_in_unit_system(unit.md_unit_system)
        new_positions = np.concatenate((positions, new_position[np.newaxis,...])) * unit.nanometer
        old_positions = new_positions

    # # minimize the new positions
    # NOTE: this is being omitted for speed purposes, but it is true that 
    # if this bit is uncommented and minimization happens, validation is OK! at around ~0.2kcal/mol
    # context = openmm.Context(new_system, openmm.VerletIntegrator(1.))
    # context.setPositions(new_positions)
    # _ = openmm.LocalEnergyMinimizer.minimize(context)
    # new_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # del context
    

    # create hybrid system and test the energy endstates
    factory = SCRFSingleTopologyHybridSystemFactory(old_system = old_system,
                                                   new_system = new_system,
                                                   old_to_new_atom_map = old_to_new_atom_map,
                                                   unique_old_atoms = unique_old_atoms, # this is empty
                                                   unique_new_atoms = unique_new_atoms,
                                                   make_old_new_rf_systems = True)
    
    _ = factory.test_energy_endstates(old_positions, new_positions, atol = 1e-1, verbose=True)

    # return the system, pos, top
    return factory, old_positions, new_positions, new_topology


def run_collect_eq(
    run_system: openmm.System, # system that will run `openmm.integrators.LangevinMiddleIntegrator`
    init_positions: np.ndarray * unit.nanometers, # initial positions
    init_box_vectors: typing.Tuple[openmm.vec3.Vec3],
    global_param_dict: typing.Dict[str, float],
    steps_per_collection: int = 25, # number of md steps to run before collecting energies
    num_collections: int = 1000, # number of energy collections to do
    temperature: float = 300. * unit.kelvin,
    dt: float = 0.002,
    gamma: float = 1.,
    platform_name: str = 'CUDA',
    enforcePeriodicBox: bool=False, # do this to not wrap particles in the box
    init_equilibration_iters: int = 10000, # 20ps
    **unused_kwargs) -> typing.Union[openmm.Context, typing.Tuple[np.ndarray, np.ndarray]]:
    """run equilibrium simulation of a system; returns positions and box vectors from every save iteration;
    if the positions and/or box vectors cannot be queries (e.g. particle coords are `NaN`, return the context for investigation)
    """

    # make platform, integrator, context
    platform = make_platform(platform_name)
    integrator = openmm.LangevinMiddleIntegrator(temperature, gamma, dt)
    context = openmm.Context(run_system, 
                             integrator,
                             platform)
    _ = [context.setParameter(name, val) for name, val in global_param_dict.items()]
    context.setPositions(init_positions)
    context.setPeriodicBoxVectors(*init_box_vectors)
    context.setVelocitiesToTemperature(temperature)

    # minimize
    print(f"minimizing...")
    openmm.LocalEnergyMinimizer.minimize(context)
    
    # equilibrate
    print(f"equilibrating...")
    integrator.step(init_equilibration_iters)
    

    out_positions, out_bvs = [], []
    print(f"running production...")
    for _iter in tqdm.trange(num_collections):
        #print(f"iter: {_iter}")
        try:
            integrator.step(steps_per_collection)
            state = context.getState(getPositions=True, enforcePeriodicBox=enforcePeriodicBox)
            positions = state.getPositions(asNumpy=True)
            out_positions.append(positions.value_in_unit_system(unit.md_unit_system))
            out_bvs.append(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit_system(unit.md_unit_system))
        except Exception as e:
            return context
        
    # garbage
    del context
    del integrator
    
    return np.array(out_positions), np.array(out_bvs)

def run_collect_neq(
    run_system: openmm.System, # system that will run integrator
    indices_to_query: jax.Array,
    fixed_site_idx: int,
    init_positions_bvs_cache: typing.Tuple[np.ndarray, np.ndarray], # initial positions, bvs
    checkpoint_file_prefix: str,
    steps_per_neq_collection: int = 25,
    nsteps_neq = 25000,
    temperature: float = 300. * unit.kelvin,
    dt: float = 0.002,
    gamma: float = 1.,
    platform_name: str = 'CUDA',
    forward: bool=True,
    enforcePeriodicBox: bool=True, # what we need to compute radial distances
    measure_shadow_work: bool=False, # probably unnecessary
    checkpoint_interval: int = 10,
    instantaneous_global_params: typing.Union[None, typing.Dict] = None, 
    # ^ params of instantaneous switch (if that is the case; e.g. {'lambda_global': [0.,1.], 'retain_uniques': [1,1]})
    **unused_kwargs) -> typing.List:
    """function to query a cache of equilibrium samples and do nonequilibrium sampling with them"""
    assert nsteps_neq % steps_per_neq_collection == 0 # the number of neq steps must be divisible by the number of neq collection steps
    num_neq_iters = nsteps_neq // steps_per_neq_collection # already presumed to be evenly divisible
    filename = f"{checkpoint_file_prefix}.npz" # filename for checkpoint
    
    # determine num sims to run
    eq_posits, eq_bvs = init_positions_bvs_cache
    num_collections = eq_posits.shape[0]
    assert eq_posits.shape[0] == eq_bvs.shape[0] # the eq cache must have corresponding positions,bvs

    # determine lambdas; no need to call `setParameter` on context
    instantaneous = instantaneous_global_params is not None
    if not instantaneous:
        alchemical_functions = {'lambda_global': 'lambda'} if forward else {'lambda_global': '1 - lambda'}
        global_params = {} # unused
    else:
        alchemical_functions = {} # unused
        global_params = instantaneous_global_params

    # make neq integrator, context
    if instantaneous: # there is no alchemical integrator
        neq_integrator = openmm.LangevinMiddleIntegrator(temperature, gamma, dt)
    else:
        neq_integrator = openmmtools.integrators.AlchemicalNonequilibriumLangevinIntegrator(
            temperature = temperature,
            collision_rate = gamma / unit.picoseconds,
            timestep = dt * unit.femtosecond,
            alchemical_functions = alchemical_functions,
            splitting = "V R H O R V", # think that is right.
            nsteps_neq = nsteps_neq,
            measure_shadow_work = measure_shadow_work)

    platform = make_platform(platform_name)
    neq_context = openmm.Context(run_system, neq_integrator, platform)

    # make collectors
    position_collector, velocity_collector, work_collector = [], [], []
    
    for _iter in tqdm.trange(num_collections):
        start_pos, start_bvs = eq_posits[_iter], eq_bvs[_iter]
        if not instantaneous:
            neq_integrator.reset() # always reset at start of new run
        else: # set the original (old) params if instantaneous
            # these will be switches to new terms before integration...
            _ = [neq_context.setParameter(name, val[0]) for name, val in global_params.items()]

        neq_context.setPositions(start_pos)
        neq_context.setPeriodicBoxVectors(*start_bvs)
        neq_context.setVelocitiesToTemperature(temperature) # randomize velocities once before all neq integration.
        
        # first, grab eq rs with `enforcePeriodicBox=True` to record before run
        eq_state = neq_context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=enforcePeriodicBox)
        _eq_posits, _eq_velocities, _eq_bvs = (eq_state.getPositions(asNumpy=True), 
            eq_state.getVelocities(asNumpy=True),
            eq_state.getPeriodicBoxVectors(asNumpy=True) )
        eq_rs, eq_vs = compute_radial_distances_and_velocities(
            positions = _eq_posits.value_in_unit_system(unit.md_unit_system), 
            velocities = _eq_velocities.value_in_unit_system(unit.md_unit_system),
            indices_to_query = indices_to_query, 
            fixed_site_idx = fixed_site_idx)
        
        if instantaneous: # set to final params and get work first
            start_energy = eq_state.getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
            _ = [neq_context.setParameter(name, val[1]) for name, val in global_params.items()]
            final_energy = neq_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
            _work_collector = [final_energy - start_energy]
        else: # there is no 'work' to get...
            _work_collector = [0.]


        _position_collector, _velocity_collector, _work_collector = [eq_rs], [eq_vs], _work_collector # inner collector
        
        # now run neq
        failed = False # the neq run has not failed
        for __iter in range(num_neq_iters):
            try:
                neq_integrator.step(steps_per_neq_collection)
                neq_state = neq_context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=enforcePeriodicBox)
                neq_pos = neq_state.getPositions(asNumpy=True)
                neq_vels = neq_state.getVelocities(asNumpy=True)
                _rs, _vs = compute_radial_distances_and_velocities(
                    positions = neq_pos.value_in_unit_system(unit.md_unit_system), 
                    velocities = neq_vels.value_in_unit_system(unit.md_unit_system),
                    indices_to_query = indices_to_query, 
                    fixed_site_idx = fixed_site_idx)
                
                if instantaneous: # there is no prot work.
                    prot_work = 0.
                else:
                    prot_work = neq_integrator.protocol_work.value_in_unit_system(unit.md_unit_system)

                # append appropriate data to inner collectors
                _position_collector.append(_rs)
                _velocity_collector.append(_vs)
                _work_collector.append(prot_work)
            except Exception as e:
                print(f"neq exception: {e}; omitting this run")
                failed=True
                break # break from neq iteration loop with failed=True

        if not failed: # only add the neq data if it has not failed...
            position_collector.append(_position_collector)
            velocity_collector.append(_velocity_collector)
            work_collector.append(_work_collector)

        if _iter == 0: # save first and reset the outter collectors
            pc, vc, wc = np.array(position_collector), np.array(velocity_collector), np.array(work_collector)
            print(f"saving data to {filename}...")
            np.savez(filename, positions = pc, velocities = vc, works = wc)
            position_collector, velocity_collector, work_collector = [], [], [] # reset outter collector
        elif (_iter + 1) % checkpoint_interval == 0: # have to add 1 because indexing starts at 0
            pc, vc, wc = np.array(position_collector), np.array(velocity_collector), np.array(work_collector)
            print(f"saving data to {filename}...")
            data = np.load(filename)
            d_pc, d_vc, d_wc = data['positions'], data['velocities'], data['works']
            out_pc = np.concatenate([d_pc, pc])
            out_vc = np.concatenate([d_vc, vc])
            out_wc = np.concatenate([d_wc, wc])
            np.savez(filename, positions = out_pc, velocities = out_vc, works = out_wc)
            position_collector, velocity_collector, work_collector = [], [], [] # reset outter collector
        
    # garbage
    del neq_context
    del neq_integrator