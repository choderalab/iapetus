"""
Test for the porinmembranesystem module.
"""

# Import packages
import os
import iapetus
import simtk.openmm as mm
from simtk.openmm import unit
import simtk.openmm.app as app
from iapetus.porin_membrane_system import PorinMembraneSystem
from simtk.openmm.app import PDBFile, ForceField

def test_porinmembranesystem(capsys):
    """Test the addition of a ligand to a solvated porin"""
    # pdb file corresponding to a solvated lipid molecule
    pdb = PDBFile(os.path.join(os.path.dirname(__file__), '../data/solvated-porin.pdb'))
    modeller = app.Modeller(pdb.topology,pdb.positions)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    platform = mm.Platform.getPlatformByName('CUDA')
    modeller.addHydrogens(forcefield=forcefield)
    system_md = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     rigidWater=False,
                                     nonbondedCutoff=1*unit.nanometer)
    system = PorinMembraneSystem('comp7', system_md, modeller.topology, modeller.positions, platform)
    integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2*unit.femtosecond)

    simulation = app.Simulation(system.structure.topology, system.backup_system, integrator, platform)
    simulation.context.setPositions(system.structure.positions)
    state = simulation.context.getState(getEnergy=True)

    print("energy is",state.getPotentialEnergy()._value)
