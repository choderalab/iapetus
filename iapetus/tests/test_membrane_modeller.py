"""
Test for the membrane_modeller module.
"""

# Import packages
import os
import iapetus
import simtk.openmm as mm
from simtk.openmm import unit
import simtk.openmm.app as app
from iapetus.membrane_modeller import MembraneModeller
from simtk.openmm.app import PDBFile, ForceField

def test_membrane_modeller():
    """Test the addition of hydrogens to a solvated DPPC molecule"""
    # pdb file corresponding to a solvated lipid molecule
    pdb = PDBFile(os.path.join(os.path.dirname(__file__), '../data/dppc/solvated-dppc.pdb'))
    modeller = MembraneModeller(pdb.topology,pdb.positions)
    modeller.modify_topology()
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller.addHydrogens(forcefield=forcefield)

    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     rigidWater=True,
                                     nonbondedCutoff=1*unit.nanometer)
    integrator = mm.VerletIntegrator(0.5*unit.femtoseconds)
    platform = mm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
    # Minimize the system after adding hydrogens
    simulation.minimizeEnergy(maxIterations=200)
    # Run a few MD steps to check the system has no overlaps
    simulation.step(1000)
    state = simulation.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy()._value
    assert pe < 0.0
