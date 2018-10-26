"""
Test for the porin_membrane_system module.
"""

# Import packages
import os
import iapetus
import simtk.openmm as mm
from simtk.openmm import unit
import simtk.openmm.app as app
from iapetus.porin_membrane_system import PorinMembraneSystem
from simtk.openmm.app import PDBFile, ForceField

def test_porin_membrane_system():
    """Test the addition of a ligand to a solvated porin"""
    # pdb file corresponding to a solvated porin
    pdb = PDBFile(os.path.join(os.path.dirname(__file__), '../data/porin/solvated-porin.pdb'))
    modeller = app.Modeller(pdb.topology,pdb.positions)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    platform = mm.Platform.getPlatformByName('CPU')
    modeller.addHydrogens(forcefield=forcefield)
    # rigidWater False is required for ParMed to access water paramters
    system_md = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     rigidWater=False,
                                     nonbondedCutoff=1*unit.nanometer)
    ligand_system = PorinMembraneSystem('comp7', system_md, modeller.topology, modeller.positions, platform, max_iterations=200)
    integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2*unit.femtosecond)
    simulation = app.Simulation(ligand_system.structure.topology, ligand_system.system, integrator, platform)
    simulation.context.setPositions(ligand_system.structure.positions)
    state = simulation.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy()._value
    assert -300000.0 <= pe <= -200000.0
