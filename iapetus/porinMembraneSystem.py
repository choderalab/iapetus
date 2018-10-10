# ==============================================================================
# FILE DOCSTRING
# ==============================================================================

"""

Porin Membrane System
=====================

Place a ligand in the porin-membrane system using ParMed Structure_.
Freeze the ligand-porin-membrane atoms and minimize the system energy by applying
the OpenMM minimizeEnergy_ method only to the water molecules and ions.

.. _Structure: https://parmed.github.io/ParmEd/html/api/parmed/parmed.structure.html?highlight=structure#parmed.structure.Structure
.. _minimizeEnergy: http://docs.openmm.org/latest/userguide/application.html?highlight=modeller#energy-minimization

"""

import os
import mdtraj as md
import parmed as pmd
import simtk.openmm as mm
import simtk.unit as unit
from copy import deepcopy
import simtk.openmm.app as app
  # Use the FIRE minimizer
from yank.fire import FIREMinimizationIntegrator


class PorinMembraneSystem(object):
    """
    Add a ligand to the porin-membrane system using ParMed_ and get the resulting
    Structure_.
    Place the ligand at the geometrical center of the porin's bottom.
    Freeze the ligand-porin-membrane atoms and minimize the system energy by
    applying the OpenMM minimizeEnergy_ method only to the water molecules and ions.

    .. _ParMed: https://parmed.github.io/ParmEd/html/index.html
    .. _Structure: https://parmed.github.io/ParmEd/html/api/parmed/parmed.structure.html?highlight=structure#parmed.structure.Structure
    .. _minimizeEnergy: http://docs.openmm.org/latest/userguide/application.html?highlight=modeller#energy-minimization

    """

    def __init__(self, ligand_name, system, topology, positions, platform, membrane=None):

        """
        Parameters
        ----------
        ligand_name : str
            The name of the ligand
        system : object
            An OpenMM system
        topology : object
            An OpenMM topology
        positions : list
            Positions stored in an OpenMM Context_
        platform : object
            Platform used by OpenMM
        membrane : str, optional, default=None
            The name of the membrane


        Attributes
        ----------
        ligand : str
            The residue name assigned to the ligand in the AMBER parameter/topology file (.prmtop)
        structure : object
            ParMed Structure_ of the minimized ligand-porin-membrane system

        .. _Structure : https://parmed.github.io/ParmEd/html/api/parmed/parmed.structure.html?highlight=structure#parmed.structure.Structure
        .. _Context : http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html#simtk.openmm.openmm.Context
        """

        # Retrieve the residue name of the ligand from the AMBER parameter/topology
        # file (.prmtop)
        with open(os.path.join(os.path.dirname(__file__),
                               'data/comp7_amber', ligand_name + '.prmtop')) as f:
            _res_label = False
            for line in f:
                if (_res_label):
                    self.ligand = str(next(f).split()[0])
                    break
                if line.startswith('%'+'FLAG RESIDUE_LABEL'):
                    _res_label = True

        structure =  pmd.openmm.load_topology(topology,
                                              system=system,
                                              xyz=positions)

        ligand_structure = pmd.load_file(os.path.join(os.path.dirname(__file__),
                                         'data/comp7_amber', ligand_name + '.prmtop'),
                                         xyz=os.path.join(os.path.dirname(__file__),
                                         'data/comp7_amber', ligand_name + '.inpcrd'))
        # Save porin indices
        top = md.Topology.from_openmm(topology)
        if membrane is not None:
            porin_indices = top.select('(protein and not resname ' + membrane + ')')
        else:
            porin_indices = top.select('protein')
        # Place the ligand at the geometrical center of the porin's bottom and get
        # the ParMed Structure
        self.structure = self._place_ligand(structure, ligand_structure, porin_indices)
        # Select atoms to be freezed during the minimization
        top = md.Topology.from_openmm(self.structure.topology)
        if membrane is not None:
            atoms_to_freeze = top.select('protein or resname  ' + membrane + ' or resname ' + self.ligand)
        else:
            atoms_to_freeze = top.select('protein or resname ' + self.ligand)
        # Perform the minimization of the ligand-porin-membrane
        self._minimize_energy(atoms_to_freeze, platform)

    def _place_ligand(self, structure, ligand_structure, porin_indices):

        """
        Place the ligand at the geometrical center of the porin's bottom and
        get the resulting ligand-porin-membrane ParMed Structure.

        Parameters
        ----------
        structure : object
            The porin-membrane system ParMed Structure
        ligand-structure : object
            The ligand ParMed structure
        porin_indices : list
            List of indices corresponding to the porin atoms

        Returns
        ----------
            The ligand-porin-membrane ParMed Structure

        """

        coords = structure.get_coordinates(frame=0)
        coords_ligand = ligand_structure.get_coordinates(frame=0)
        coords_ligand += coords[porin_indices,:].mean(0) - coords_ligand[:,:].mean(0)
        coords_ligand[:,2] += coords[porin_indices,:].min(0)[2] - coords[porin_indices,:].mean(0)[2]
        ligand_structure.coordinates = coords_ligand
        new_structure = structure + ligand_structure

        return new_structure

    def _minimize_energy(self, atoms_to_freeze, platform):
        """
        Use the OpenMM minimizeEnergy method in a system with the
        ligand-porin-membrane atoms frezeed. Only the water molecules and ions
        will moved to avoid overlaps with the ligand.

        Parameters
        ----------
        atoms_to_freeze : list
            List of atoms that won't move during minimization
        platform : object
            Platform used by OpenMM

        """

        system = self.structure.createSystem(nonbondedMethod=app.PME,
                                 nonbondedCutoff=1*unit.nanometer,
                                 rigidWater=True,
                                 flexibleConstraints=True,
                                 constraints=app.HBonds,
                                 hydrogenMass=3*unit.amu,
                                 removeCMMotion=False)
        self.backup_system =  deepcopy(system)
        # zero mass = atoms won't move
        for index in atoms_to_freeze:
            system.setParticleMass(int(index), 0.0*unit.dalton)
        integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2*unit.femtosecond)
        simulation = app.Simulation(self.structure.topology, system, integrator, platform)
        simulation.context.setPositions(self.structure.positions)
        simulation.minimizeEnergy(tolerance=0.1*unit.kilojoule/unit.mole, maxIterations=2000)
        self.structure.positions = context.getState(getPositions=True).getPositions()
        del simulation.context, integrator
