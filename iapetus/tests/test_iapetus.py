"""
Unit and regression test for the iapetus package.
"""

# Import package, test suite, and other packages as needed
import os
import sys
import pytest
import errno
import shutil
import tempfile
from pkg_resources import resource_filename

from simtk import unit

import iapetus

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the openmoltools folder).
    """

    fn = resource_filename('iapetus', os.path.join('data/gromacs', relative_path))
    print(fn)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn

def test_iapetus_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "iapetus" in sys.modules

def test_gromacs():
    """Test that gromacs input data can be processed"""
    from iapetus import SimulatePermeation
    from iapetus import GromacsSystem
    from iapetus import PlatformSettings
    from iapetus import LoggerSettings

    gromacs_input_path = get_data_filename('comp7_nowat/')
    ligand_resseq = 422
    tmp_dir = tempfile.mkdtemp()
    output_filename = os.path.join(tmp_dir, 'output.nc')
    platform_settings = PlatformSettings(platform_name='CPU')
    LoggerSettings(verbose=True)
    membrane = 'None'
    system = GromacsSystem('../data','gromacs', 'comp7_nowat', ligand_resseq=ligand_resseq, membrane=membrane)
    positions = system.get_positions(platform=platform_settings.platform)
    topology = system.get_topology()
    box = system.get_box()
    ligand_resseq = system.get_ligand_resseq()
    simulation = SimulatePermeation(topology, ligand_resseq=ligand_resseq, membrane=membrane, output_filename=output_filename)
    simulation.pressure = None
    simulation.n_iterations = 2
    simulation.n_steps_per_iteration = 5
    simulation.timestep = 2.0 * unit.femtoseconds
    simulation.anneal_ligand = True
    openmm_system = system.create_system(pressure=simulation.pressure)
    simulation.run(openmm_system, topology, positions, box, resume=False)
    shutil.rmtree(tmp_dir)

def test_cli():
    """Test the CLI"""
    from iapetus import main
    gromacs_input_path = get_data_filename('comp7_nowat/')
    tmp_dir = tempfile.mkdtemp()
    output_filename = os.path.join(tmp_dir, 'output.nc')
    sys.argv = ["prog", "--path", "../data", "--source", "gromacs", "--ligand_name", "comp7_nowat", "--ligseq", "422", "--output", output_filename, "--niterations", "2", '--platform', 'CPU', '--ncontexts', '3', '--testmode', '--n_steps_per_iteration', '5']
    main()
    shutil.rmtree(tmp_dir)
