"""
Unit and regression test for the iapetus package.
"""

# Import package, test suite, and other packages as needed
import iapetus
import pytest
import sys
import os
from pkg_resources import resource_filename

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the openmoltools folder).
    """

    fn = resource_filename('iapetus', os.path.join('data', relative_path))
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
    gromacs_input_path = get_data_filename('arg/')
    ligand_resseq = 423
    output_filename = 'output.nc' # TODO: Use tmpfile
    simulation = SimulatePermeation(gromacs_input_path=gromacs_input_path, ligand_resseq=ligand_resseq, output_filename=output_filename)
    simulation.setup()
    simulation.run(n_iterations=2)

def test_cli():
    """Test the CLI"""
    from iapetus import main
    gromacs_input_path = get_data_filename('arg/')
    # TODO: use tmpfile for output.nc
    sys.argv = ["prog", "--gromacs", gromacs_input_path, "--ligseq", "423", "--output", "output.nc", "--niterations", "2"]
    main()
