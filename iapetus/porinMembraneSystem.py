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

import mdtraj as md
import parmed as pmd
import simtk.openmm as mm
import simtk.unit as unit
import simtk.openmm.app as app

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
        positions : array
            Positions stored in an OpenMM context
        platform : object
            Platform used by OpenMM
        membrane : str
            The name of the membrane


        Attributes
        ----------
        ligand : str
            The residue name assigned to the ligand in the AMBER parameter/topology file (.prmtop)
        structure : object
            ParMed Structure_ of the minimized ligand-porin-membrane system

        .. _Structure : .. _Structure: https://parmed.github.io/ParmEd/html/api/parmed/parmed.structure.html?highlight=structure#parmed.structure.Structure

        """

        # Retrieve the residue name of the ligand from the AMBER parameter/topology
        # file (.prmtop)
        with open(ligand_name + '.prmtop') as f:
            res_label = False
            for line in f:
                if (res_label):
                    self.ligand = str(next(f).split()[0])
                    break
                if line.startswith('%'+'FLAG RESIDUE_LABEL'):
                    res_label = True

        structure =  pmd.openmm.load_topology(topology,
                                              system=system,
                                              xyz=positions)
        ligand_structure = pmd.load_file(ligand_name + '.prmtop', xyz=ligand_name + '.inpcrd')
        # Save porin indices
        top = md.Topology.from_openmm(topology)
        if membrane is not None:
            porin_indices = topo.select('(protein and not resname ' + membrane + ')')
        else:
            porin_indices = topo.select('protein')
        # Place the ligand at the geometrical center of the porin's bottom and get
        # the ParMed Structure
        self.structure = self._PlaceLigand(structure, ligand_structure, porin_indices)
        # Select atoms to be freezed during the minimization
        top = md.Topology.from_openmm(self.structure.topology)
        atoms_to_freeze = new_topo.select('protein or resname  ' + membrane + ' or resname ' + self.ligand)
        # Perform the minimization of the ligand-porin-membrane
        self._MinimizeEnergy(atoms_to_freeze, platform)

    def _PlaceLigand(self, structure, ligand_structure, porin_indices):

        """
        Place the ligand at the geometrical center of the porin's bottom.
        Get the  ligand-porin-membrane ParMed Structure_.

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

    def _MinimizeEnergy(self, atoms_to_freeze, platform):
        """
        Use the OpenMM minimizeEnergy_ method in a system with the
        ligand-porin-membrane atoms frezeed. Only the water molecules and ions
        will moved to avoid overlaps with the ligand.


        Parameters
        ----------
        atoms_to_freeze : List
            List of atoms that won't move during minimization
        platform : object
            Platform used by OpenMM

        .. _minimizeEnergy: http://docs.openmm.org/latest/userguide/application.html?highlight=modeller#energy-minimization

        """

        system = self.structure.createSystem(nonbondedMethod=app.PME,
                                 nonbondedCutoff=1*unit.nanometer,
                                 constraints=app.HBonds,
                                 hydrogenMass=3*unit.amu,
                                 removeCMMotion=False)
        # zero mass = atoms won't move
        for index in atom_to_freeze:
            system.setParticleMass(int(index), 0.0*unit.dalton)
        integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2*unit.femtosecond)
        simulation = app.Simulation(self.structure.topology, system, integrator, platform)
        simulation.context.setPositions(self.structure.positions)
        simulation.minimizeEnergy(tolerance=0.1*unit.kilojoule/unit.mole, maxIterations=2000)
        positions = simulation.context.getState(getPositions=True).getPositions()
        del context, integrator
