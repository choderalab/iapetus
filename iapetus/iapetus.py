"""
iapetus.py
Open source toolkit for predicting bacterial porin permeation

Handles the primary functions
"""

import os
import copy
import time
import pathlib
import logging
import argparse

import progressbar

import mdtraj as md
import numpy as np

import yank

from simtk import openmm, unit
from simtk.openmm import app

import openmmtools
from openmmtools.constants import kB
from openmmtools import integrators, states, mcmc

logger = logging.getLogger(__name__)

class SimulatePermeation(object):
    """

    Properties
    ----------
    anneal_ligand : bool, optional, default=True
        If True, will relax the ligand by alchemically annealing it first.
        Recommended, but can be slow on CPUs.

    """
    def __init__(self, gromacs_input_path=None, ligand_resseq=None, output_filename=None, verbose=False):
        """Set up a SAMS permeation PMF simulation.

        Parameters
        ----------
        gromacs_input_path : str, optional, default=None
            If specified, use gromacs files in this directory
        ligand_resseq : str, optional, default='MOL'
            Resdiue sequence id for ligand in reference PDB file
        output_filename : str, optional, default=None
            NetCDF output filename
        verbose : bool, optional, default=False
            If True, print verbose output

        """
        self._setup_complete = False

        # Setup general logging
        logging.root.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        yank.utils.config_root_logger(verbose=verbose, log_file_path=None)

        # Set default parameters
        self.temperature = 310.0 * unit.kelvin
        self.pressure = 1.0 * unit.atmospheres
        self.collision_rate = 1.0 / unit.picoseconds
        self.timestep = 2.0 * unit.femtoseconds
        self.n_steps_per_iteration = 1250
        self.n_iterations = 10000
        self.checkpoint_interval = 50
        self.gamma0 = 10.0
        self.flatness_threshold = 10.0
        self.anneal_ligand = True

        # Check input
        if gromacs_input_path is None:
            raise ValueError('gromacs_input_path must be specified')
        if ligand_resseq is None:
            raise ValueError('ligand_resseq must be specified')
        if output_filename is None:
            raise ValueError('output_filename must be specified')

        # Discover contents of the input path by suffix
        contents = { pathlib.Path(filename).suffix : filename for filename in os.listdir(gromacs_input_path) }
        gro_filename = os.path.join(gromacs_input_path, contents['.gro'])
        top_filename = os.path.join(gromacs_input_path, contents['.top'])
        pdb_filename = os.path.join(gromacs_input_path, contents['.pdb'])

        # Load system files
        print('Reading system from path: {}'.format(gromacs_input_path))
        self.grofile = app.GromacsGroFile(gro_filename)
        self.topfile = app.GromacsTopFile(top_filename, periodicBoxVectors=self.grofile.getPeriodicBoxVectors())
        self.pdbfile = app.PDBFile(pdb_filename)

        # Create MDTraj Trajectory for reference PDB file for use in atom selections and slicing
        self.mdtraj_refpdb = md.load(pdb_filename)
        self.mdtraj_topology = self.mdtraj_refpdb.topology
        self.analysis_particle_indices = self.mdtraj_topology.select('not water')

        # Store output filename
        # TODO: Check if file already exists and resume if so
        self.output_filename = output_filename

        # Store ligand resseq
        self.ligand_resseq = ligand_resseq

    def _setup(self):
        """
        Set up calculation.

        """
        # Signal that system has now been set up
        if self._setup_complete:
            raise Exception("System has already been set up---cannot run again.")
        self._setup_complete = True

        # Compute thermal energy and inverse temperature
        self.kT = kB * self.temperature
        self.beta = 1.0 / self.kT

        # Create the system
        self.system = self._create_system()

        # Add a barostat
        # TODO: Is this necessary, sicne ThermodynamicState handles this automatically? It may not correctly handle MonteCarloAnisotropicBarostat.
        self._add_barostat()

        # Restrain protein atoms in space
        # TODO: Allow protein atom selection to be configured
        selection = '((residue 342 and resname GLY) or (residue 97 and resname ASP) or (residue 184 and resname SER)) and (name CA)'
        protein_atoms_to_restrain = self.mdtraj_topology.select(selection)
        #self._restrain_protein(protein_atoms_to_restrain)

        # Create SamplerState for initial conditions
        self.sampler_state = states.SamplerState(positions=self.grofile.positions, box_vectors=self.grofile.getPeriodicBoxVectors())

        # Create reference thermodynamic state
        self.reference_thermodynamic_state = states.ThermodynamicState(system=self.system, temperature=self.temperature, pressure=self.pressure)

        # Anneal ligand into binding site
        if self.anneal_ligand:
            self._anneal_ligand()

        # Create ThermodynamicStates for umbrella sampling along pore
        self.thermodynamic_states = self._create_thermodynamic_states(self.reference_thermodynamic_state)

        # Minimize initial thermodynamic state
        # TODO: Select initial thermodynamic state based on which state has minimum energy
        initial_state_index = 0
        self.sampler_state = self._minimize_sampler_state(self.thermodynamic_states[initial_state_index], self.sampler_state)

        # Set up simulation
        from yank.multistate import SAMSSampler, MultiStateReporter
        move = mcmc.LangevinDynamicsMove(timestep=self.timestep, collision_rate=self.collision_rate, n_steps=self.n_steps_per_iteration, reassign_velocities=False)
        self.simulation = SAMSSampler(mcmc_moves=move, number_of_iterations=self.n_iterations, online_analysis_interval=None, gamma0=self.gamma0, flatness_threshold=self.flatness_threshold)
        self.reporter = MultiStateReporter(self.output_filename, checkpoint_interval=self.checkpoint_interval, analysis_particle_indices=self.analysis_particle_indices)
        self.simulation.create(thermodynamic_states=self.thermodynamic_states,
                               unsampled_thermodynamic_states=[self.reference_thermodynamic_state],
                               sampler_states=[self.sampler_state], initial_thermodynamic_states=[initial_state_index],
                               storage=self.reporter)

    def run(self, platform_name=None, precision='auto', max_n_contexts=None):
        """
        Run the sampler for a specified number of iterations

        Parameters
        ----------
        platform_name : str, optional, default=None
            Name of platform, or 'fastest' if fastest platform should be automatically selected
        precision : str, optional, default='auto'
            Precision to use, or None to automatically select ('mixed' is used if supported)
        max_n_contexts : int, optional, default=None
            Maximum number of contexts to use

        """
        # Configure ContextCache, platform and precision
        from yank.experiment import ExperimentBuilder
        platform = ExperimentBuilder._configure_platform(platform_name, precision)

        try:
            openmmtools.cache.global_context_cache.platform = platform
        except RuntimeError:
            # The cache has been already used. Empty it before switching platform.
            openmmtools.cache.global_context_cache.empty()
            openmmtools.cache.global_context_cache.platform = platform

        if max_n_contexts is not None:
            openmmtools.cache.global_context_cache.capacity = max_n_contexts

        # Set up the simulation if it has not yet been set up
        if not self._setup_complete:
            self._setup()

        # Run the simulation
        self.simulation.run()

    def _create_thermodynamic_states(self, reference_thermodynamic_state, spacing=0.25*unit.angstroms):
        """
        Create thermodynamic states for sampling along pore axis.

        Here, the porin is restrained in the (x,z) plane and the pore axis is oriented along the y-axis.

        Parameters
        ----------
        reference_thermodynamic_state : openmmtools.states.ThermodynamicState
            Reference ThermodynamicState containing system, temperature, and pressure
        spacing : simtk.unit.Quantity with units compatible with angstroms
            Spacing between umbrellas spanning pore axis
        """
        # Determine relevant coordinates
        # TODO: Allow selections to be specified as class methods
        axis_center_atom = self.mdtraj_topology.select('residue 128 and resname ALA and name CA')[0]
        axis_center = self._get_reference_coordinates(axis_center_atom)
        print('axis center (x,z) : {} : {}'.format(axis_center_atom, axis_center))
        pore_top_atom = self.mdtraj_topology.select('residue 226 and resname SER and name CA')[0]
        pore_top = self._get_reference_coordinates(pore_top_atom)
        print('pore top (y-max): {} : {}'.format(pore_top_atom, pore_top))
        pore_bottom_atom = self.mdtraj_topology.select('residue 88 and resname VAL and name CA')[0]
        pore_bottom = self._get_reference_coordinates(pore_bottom_atom)
        print('pore bottom (y-min): {} : {}'.format(pore_bottom_atom, pore_bottom))

        # Determine ligand atoms
        selection = '(residue {}) and (mass > 1.5)'.format(self.ligand_resseq)
        print('Determining ligand atoms using "{}"...'.format(selection))
        ligand_atoms = self.mdtraj_topology.select(selection)
        if len(ligand_atoms) == 0:
            raise ValueError('Ligand residue name {} not found'.format(self.ligand_resseq))
        print('ligand heavy atoms: {}'.format(ligand_atoms))

        # Determine protein atoms
        bottom_selection = '((residue 342 and resname GLY) or (residue 97 and resname ASP) or (residue 184 and resname SER)) and (name CA)'
        bottom_protein_atoms = self.mdtraj_topology.select(bottom_selection)
        top_selection = '((residue 226 and resname SER) or (residue 421 and resname LEU) or (residue 147 and resname GLU)) and (name CA)'
        top_protein_atoms = self.mdtraj_topology.select(top_selection)

        # Compute pore bottom (lambda=0) and top (lambda=1)
        axis_bottom = (axis_center[0], pore_bottom[1], axis_center[2])
        axis_top = (axis_center[0], pore_top[1], axis_center[2])
        axis_distance = pore_top[1] - pore_bottom[1]
        print('axis_distance: {}'.format(axis_distance))

        # Compute spacing and spring constant
        expansion_factor = 1.2
        nstates = int(expansion_factor * axis_distance / spacing) + 1
        print('nstates: {}'.format(nstates))
        sigma_y = axis_distance / float(nstates) # stddev of force-free fluctuations in y-axis
        K_y = self.kT / (sigma_y**2) # spring constant
        print('veritcal sigma_y = {:.3f} A'.format(sigma_y / unit.angstroms))

        # Compute restraint width
        # TODO: Come up with a better way to define pore width?
        scale_factor = 0.25
        sigma_xz = scale_factor * unit.sqrt( (axis_center[0] - pore_bottom[0])**2 + (axis_center[2] - pore_bottom[2])**2 )  # stddev of force-free fluctuations in xz-plane
        K_xz = self.kT / (sigma_xz**2) # spring constant
        print('in-plane sigma_xz = {:.3f} A'.format(sigma_xz / unit.angstroms))

        dr = axis_distance * (expansion_factor - 1.0)/2.0
        rmax = axis_distance + dr
        rmin = - dr

        # Create restraint state that encodes this axis
        # TODO: Rework this as CustomCVForce with in-plane and along-axis deviations as separate forces so we can store them each iteration
        print('Creating restraint...')
        from yank.restraints import RestraintState
        energy_expression = '(K_parallel/2)*(r_parallel-r0)^2 + (K_orthogonal/2)*r_orthogonal^2;'
        energy_expression += 'r_parallel = r*cos(theta);'
        energy_expression += 'r_orthogonal = r*sin(theta);'
        energy_expression += 'r = distance(g1,g2);'
        energy_expression += 'theta = angle(g1,g2,g3);'
        energy_expression += 'r0 = lambda_restraints * (rmax - rmin) + rmin;'
        force = openmm.CustomCentroidBondForce(3, energy_expression)
        force.addGlobalParameter('lambda_restraints', 1.0)
        force.addGlobalParameter('K_parallel', K_y)
        force.addGlobalParameter('K_orthogonal', K_xz)
        force.addGlobalParameter('rmax', rmax)
        force.addGlobalParameter('rmin', rmin)
        force.addGroup([int(index) for index in ligand_atoms])
        force.addGroup([int(index) for index in bottom_protein_atoms])
        force.addGroup([int(index) for index in top_protein_atoms])
        force.addBond([0,1,2], [])
        self.system.addForce(force)
        # Update reference thermodynamic state
        print('Updating system in reference thermodynamic state...')
        self.reference_thermodynamic_state.set_system(self.system, fix_state=True)

        # Create alchemical state
        #from openmmtools.alchemy import AlchemicalState
        #alchemical_state = AlchemicalState.from_system(self.reference_thermodynamic_state.system)

        # Create restraint state
        restraint_state = RestraintState(lambda_restraints=1.0)

        # Create thermodynamic states to be sampled
        # TODO: Should we include an unbiased state?
        initial_time = time.time()
        thermodynamic_states = list()
        #compound_state = states.CompoundThermodynamicState(self.reference_thermodynamic_state, composable_states=[alchemical_state, restraint_state])
        compound_state = states.CompoundThermodynamicState(self.reference_thermodynamic_state, composable_states=[restraint_state])
        for lambda_restraints in np.linspace(0, 1, nstates):
            thermodynamic_state = copy.deepcopy(compound_state)
            thermodynamic_state.lambda_restraints = lambda_restraints
            #thermodynamic_state.lambda_sterics = 1.0
            #thermodynamic_state.lambda_electrostatics = 1.0
            thermodynamic_states.append(thermodynamic_state)
        elapsed_time = time.time() - initial_time
        print('Creating thermodynamic states took %.3f s' % elapsed_time)

        return thermodynamic_states

    def _get_reference_coordinates(self, atom):
        return self.mdtraj_refpdb.xyz[0,atom,:] * unit.nanometers

    def _create_system(self):
        """Create the System.
        """
        # Create System
        print('Creating System...')
        # TODO: Allow these to be user-specified parameters
        if self.pressure is None:
            # We are simulating in a vacuum
            nonbonded_method = app.NoCutoff
        else:
            # We are simulating in solvent
            nonbonded_method = app.PME
        kwargs = { 'nonbondedMethod' : nonbonded_method, 'constraints' : app.HBonds, 'rigidWater' : True, 'ewaldErrorTolerance' : 1.0e-4, 'removeCMMotion' : False, 'hydrogenMass' : 3.0*unit.amu }
        system = self.topfile.createSystem(**kwargs)

        # Fix particles with zero LJ sigma
        for force in system.getForces():
            if force.__class__.__name__ == 'NonbondedForce':
                for index in range(system.getNumParticles()):
                    [charge, sigma, epsilon] = force.getParticleParameters(index)
                    if sigma / unit.nanometers == 0.0:
                        force.setParticleParameters(index, charge, 1.0*unit.angstroms, epsilon)

        print('System has {} particles'.format(system.getNumParticles()))

        return system

    def _alchemically_modify_ligand(self, reference_system):
        """
        Alchemically soften ligand.
        """
        # Determine ligand atoms
        ligand_atoms = self.mdtraj_topology.select('residue {}'.format(self.ligand_resseq))
        if len(ligand_atoms) == 0:
            raise ValueError('Ligand residue name {} not found'.format(self.ligand_resseq))
        print('ligand heavy atoms: {}'.format(ligand_atoms))

        # Create alchemically-modified system
        # TODO: Use exact PME if PME is selected
        from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion, AlchemicalState
        #factory = AbsoluteAlchemicalFactory(consistent_exceptions=False, alchemical_pme_treatment='exact')
        factory = AbsoluteAlchemicalFactory(consistent_exceptions=False, disable_alchemical_dispersion_correction=True)
        alchemical_region = AlchemicalRegion(alchemical_atoms=ligand_atoms)
        alchemical_system = factory.create_alchemical_system(reference_system, alchemical_region)

        # Soften ligand
        #alchemical_state = AlchemicalState.from_system(alchemical_system)
        #alchemical_state.lambda_sterics = 1.0
        #alchemical_state.lambda_electrostatics = 1.0
        #alchemical_state.apply_to_system(alchemical_system)

        # Store alchemical system
        return alchemical_system

    def _anneal_ligand(self):
        """Anneal ligand interactions to clean up clashes.

        """
        alchemical_system = self._alchemically_modify_ligand(self.system)

        from openmmtools.alchemy import AlchemicalState
        alchemical_state = AlchemicalState.from_system(alchemical_system)
        thermodynamic_state = states.ThermodynamicState(system=alchemical_system, temperature=self.temperature, pressure=self.pressure)
        alchemical_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state=thermodynamic_state, composable_states=[alchemical_state])

        # Initial softened minimization
        print('Minimizing softened ligand...')
        alchemical_thermodynamic_state.lambda_sterics = 0.01
        alchemical_thermodynamic_state.lambda_electrostatics = 0.0
        self.sampler_state = self._minimize_sampler_state(alchemical_thermodynamic_state, self.sampler_state)

        # Anneal
        n_annealing_steps = 1000
        integrator = openmm.LangevinIntegrator(self.temperature, 90.0/unit.picoseconds, 1.0*unit.femtoseconds)
        context, integrator = openmmtools.cache.global_context_cache.get_context(alchemical_thermodynamic_state, integrator)
        self.sampler_state.apply_to_context(context)
        print('Annealing sterics...')
        for step in progressbar.progressbar(range(n_annealing_steps)):
            alchemical_state.lambda_sterics = float(step) / float(n_annealing_steps)
            alchemical_state.lambda_electrostatics = 0.0
            alchemical_state.apply_to_context(context)
            integrator.step(1)
        print('Annealing electrostatics...')
        for step in progressbar.progressbar(range(n_annealing_steps)):
            alchemical_state.lambda_sterics = 1.0
            alchemical_state.lambda_electrostatics = float(step) / float(n_annealing_steps)
            alchemical_state.apply_to_context(context)
            integrator.step(1)
        self.sampler_state.update_from_context(context)

        # Compute the final energy of the system for logging.
        final_energy = thermodynamic_state.reduced_potential(context)
        logger.debug('final alchemical energy {:8.3f}kT'.format(final_energy))

        # Initial softened minimization
        print('Minimizing real ligand...')
        self.sampler_state = self._minimize_sampler_state(self.reference_thermodynamic_state, self.sampler_state)

    def _add_barostat(self):
        """Add a barostat to the system if one doesn't already exist.
        """
        # TODO: For membrane protein simulations, we should add an anisotropic barostat
        has_barostat = False
        for (index, force) in enumerate(self.system.getForces()):
            if force.__class__.__name__ in ['MonteCarloBarostat', 'MonteCarloAnisotropicBarostat']:
                has_barostat = True
                force_index = index

        if self.pressure is not None:
            if not has_barostat:
                print('Adding a barostat...')
                barostat = openmm.MonteCarloBarostat(self.pressure, self.temperature)
                self.system.addForce(barostat)
        else:
            if has_barostat:
                # Remove barostat
                self.system.removeForce(force_index)

    def _restrain_protein(self, protein_atoms_to_restrain):
        """
        Restrain protein atoms in system to keep protein from moving from reference position.

        Parameters
        ----------
        protein_atoms_to_restrain : list of int
            List of atom indices to restraint

        """
        # TODO: Allow sigma_protein to be configured
        print('Adding restraints to protein...')
        sigma_protein = 2.0 * unit.angstroms # stddev of fluctuations of protein atoms
        K_protein = self.kT / (sigma_protein**2) # spring constant
        energy_expression = '(K_protein/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2);'
        force = openmm.CustomExternalForce(energy_expression)
        force.addGlobalParameter('K_protein', K_protein)
        force.addPerParticleParameter('x0')
        force.addPerParticleParameter('y0')
        force.addPerParticleParameter('z0')
        for particle_index in protein_atoms_to_restrain:
            x0, y0, z0 = self._get_reference_coordinates(particle_index) / unit.nanometers
            force.addParticle(int(particle_index), [x0, y0, z0])
        self.system.addForce(force)

    @staticmethod
    def _minimize_sampler_state(thermodynamic_state, sampler_state, tolerance=None, max_iterations=100):
        """Minimize the specified sampler state at the given thermodynamic state.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state for the simulation.
        sampler_state : openmmtools.states.SamplerState
            Initial sampler state
        tolerance : unit.Quantity compatible with kilojoules_per_mole/nanometer, optional, default=1.0*unit.kilojoules_per_mole/unit.angstroms
            Minimization will be terminated when RMS force reaches this tolerance
        max_iterations : int, optional, default=100
            Maximum number of iterations for minimization.
            If 0, minimization continues until converged.

        Returns
        -------
        minimized_sampler_state : openmmtools.states.SamplerState
            Minimized sampler state

        """
        if tolerance is None:
            tolerance = 1.0*unit.kilojoules_per_mole/unit.angstroms

        # Use the FIRE minimizer
        from yank.fire import FIREMinimizationIntegrator
        reference_integrator = FIREMinimizationIntegrator(tolerance=tolerance)

        # Create context
        #context = thermodynamic_state.create_context(integrator)
        context, integrator = openmmtools.cache.global_context_cache.get_context(thermodynamic_state, reference_integrator)

        # DEBUG: Reset FIRE minimizer integrator
        print('Resetting FIRE integrator...')
        for index in range(reference_integrator.getNumGlobalVariables()):
            value = reference_integrator.getGlobalVariable(index)
            integrator.setGlobalVariable(index, value)

        # Set initial positions and box vectors.
        sampler_state.apply_to_context(context)

        # Compute the initial energy of the system for logging.
        initial_energy = thermodynamic_state.reduced_potential(context)
        logger.debug('initial energy {:8.3f}kT'.format(initial_energy))

        # Minimize energy.
        try:
            if max_iterations == 0:
                logger.debug('Using FIRE: tolerance {} minimizing to convergence'.format(tolerance))
                while integrator.getGlobalVariableByName('converged') < 1:
                    integrator.step(50)
            else:
                logger.debug('Using FIRE: tolerance {} max_iterations {}'.format(tolerance, max_iterations))
                integrator.step(max_iterations)
        except Exception as e:
            if str(e) == 'Particle coordinate is nan':
                logger.debug('NaN encountered in FIRE minimizer; falling back to L-BFGS after resetting positions')
                sampler_state.apply_to_context(context)
                openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)
            else:
                raise e

        # Get the minimized positions.
        sampler_state.update_from_context(context)

        # Compute the final energy of the system for logging.
        final_energy = thermodynamic_state.reduced_potential(context)
        logger.debug('final energy {:8.3f}kT'.format(final_energy))

        if (final_energy >= initial_energy):
            logger.debug('minimizing again since no progress was made...')
            #sampler_state.apply_to_context(context)
            initial_energy = final_energy
            logger.debug('initial energy {:8.3f}kT'.format(initial_energy))
            openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)
            final_energy = thermodynamic_state.reduced_potential(context)
            logger.debug('final energy {:8.3f}kT'.format(final_energy))
            sampler_state.update_from_context(context)

        # Clean up the integrator
        #del context, integrator

        return sampler_state

def main():
    """Set up and run a porin permeation PMF calculation.
    """

    parser = argparse.ArgumentParser(description='Compute a potential of mean force (PMF) for porin permeation.')
    parser.add_argument('--gromacs', dest='gromacs_input_path', action='store',
                        help='gromacs input path')
    parser.add_argument('--ligseq', dest='ligand_resseq', action='store',
                        help='ligand residue sequence id')
    parser.add_argument('--output', dest='output_filename', action='store', default='output.nc',
                        help='output netcdf filename (default: output.nc)')
    parser.add_argument('--niterations', dest='n_iterations', action='store', type=int, default=10000,
                        help='number of iterations to run (default: 10000)')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='if set, will turn on verbose output (default: False)')
    parser.add_argument('--platform', dest='platform', action='store', default='fastest',
                        help='OpenMM platform to use, or "fastest" to auto-select fastest platform (default: fastest)')
    parser.add_argument('--precision', dest='precision', action='store', default='auto',
                        help='OpenMM precision to use (default: auto)')
    parser.add_argument('--ncontexts', dest='max_n_contexts', action='store', type=int, default=None,
                        help='Maximum number of contexts (default: None)')
    parser.add_argument('--n_steps_per_iteration', dest='n_steps_per_iteration', action='store', type=int, default=500,
                        help='Number of timesteps per iteration (default: 500)')
    parser.add_argument('--testmode', dest='testmode', action='store_true', default=False,
                        help='Run a vacuum simulation for testing')

    args = parser.parse_args()

    # Check all required arguments have been provided
    if (args.gromacs_input_path is None) or (args.ligand_resseq is None):
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Determine ligand residue name
    ligand_resseq = args.ligand_resseq

    # Determine the path to gromacs input
    gromacs_input_path = os.path.abspath(args.gromacs_input_path)

    # Determine output filename
    output_filename = os.path.abspath(args.output_filename)

    # Set up the calculation
    # TODO: Check if output files exist first and resume if so?
    simulation = SimulatePermeation(gromacs_input_path=gromacs_input_path, ligand_resseq=ligand_resseq, output_filename=output_filename, verbose=args.verbose)
    simulation.n_iterations = args.n_iterations
    simulation.n_steps_per_iteration = args.n_steps_per_iteration

    if args.testmode:
        simulation.pressure = None
        simulation.anneal_ligand = False

    simulation.run(platform_name=args.platform, precision=args.precision, max_n_contexts=args.max_n_contexts)

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    main()
