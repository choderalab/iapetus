"""
iapetus.py
Open source toolkit for predicting bacterial porin permeation

Handles the primary functions
"""

import os
import pathlib
import argparse

import mdtraj

from simtk import openmm, unit
from simtk.openmm import app

def setup_calculation(gromacs_input_path=None, ligand_resseq=None):
    """Set up a calculationself.

    Parameters
    ----------
    gromacs_input_path : str, optional, default=None
        If specified, use gromacs files in this directory
    ligand_resseq : str, optional, default='MOL'
        Resdiue sequence id for ligand in reference PDB file

    """
    if gromacs_input_path is None:
        raise ValueError('gromacs_input_path must be specified')
    if ligand_resseq is None:
        raise ValueError('ligand_resseq must be specified')

    # Discover contents of the input path by suffix
    contents = { pathlib.Path(filename).suffix : filename for filename in os.listdir(gromacs_input_path) }
    gro_filename = os.path.join(gromacs_input_path, contents['.gro'])
    top_filename = os.path.join(gromacs_input_path, contents['.top'])
    pdb_filename = os.path.join(gromacs_input_path, contents['.pdb'])

    # Load system files
    print('Reading system from path: {}'.format(gromacs_input_path))
    grofile = app.GromacsGroFile(gro_filename)
    topfile = app.GromacsTopFile(top_filename, periodicBoxVectors=grofile.getPeriodicBoxVectors())
    pdbfile = app.PDBFile(pdb_filename)

    # Create System
    print('Creating System...')
    # TODO: Allow these to be user-specified parameters
    kwargs = { 'nonbondedMethod' : app.PME, 'constraints' : app.HBonds, 'rigidWater' : True, 'ewaldErrorTolerance' : 1.0e-4, 'removeCMMotion' : False, 'hydrogenMass' : 3.0*unit.amu }
    system = topfile.createSystem(**kwargs)
    print('System has {} particles'.format(system.getNumParticles()))

    # Determine relevant coordinates
    print('Computing atom selections...')
    mdtraj_refpdb = mdtraj.load(pdb_filename)
    mdtraj_topology = mdtraj_refpdb.topology
    def get_coordinates(atom):
        return mdtraj_refpdb.xyz[0,atom,:] * unit.nanometers

    ligand_atoms = mdtraj_topology.select('(residue {}) and (mass > 1.5)'.format(ligand_resseq))
    plane_atoms = mdtraj_topology.select('((residue 342 and resname GLY) or (residue 97 and resname ASP) or (residue 184 and resname SER)) and (name CA)')
    axis_center_atom = mdtraj_topology.select('residue 128 and resname ALA and name CA')[0]
    pore_top_atom = mdtraj_topology.select('residue 226 and resname SER and name CA')[0]
    pore_bottom_atom = mdtraj_topology.select('residue 88 and resname VAL and name CA')[0]
    print('ligand heavy atoms: {}'.format(ligand_atoms))
    print('plane atoms (x,y,z): {}'.format(plane_atoms))
    print('axis center (x,z) : {} : {}'.format(axis_center_atom, get_coordinates(axis_center_atom)))
    print('pore top (y-max): {} : {}'.format(pore_top_atom, get_coordinates(pore_top_atom)))
    print('pore bottom (y-min): {} : {}'.format(pore_bottom_atom, get_coordinates(pore_bottom_atom)))

    if len(ligand_atoms) == 0:
        raise ValueError('Ligand residue name {} not found'.format(ligand_resseq))

def main():
    """Set up and run a porin permeation PMF calculation.
    """

    parser = argparse.ArgumentParser(description='Compute a potential of mean force (PMF) for porin permeation.')
    parser.add_argument('--gromacs', dest='gromacs_input_path', action='store',
                        help='gromacs input path')
    parser.add_argument('--ligseq', dest='ligand_resseq', action='store',
                        help='ligand residue sequence id')
    args = parser.parse_args()

    # Determine ligand residue name
    ligand_resseq = args.ligand_resseq

    # Determine the path to gromacs input
    gromacs_input_path = os.path.abspath(args.gromacs_input_path)

    # Set up the calculation
    # TODO: Check if output files exist first and resume if so?
    setup_calculation(gromacs_input_path=gromacs_input_path, ligand_resseq=ligand_resseq)

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    main()
