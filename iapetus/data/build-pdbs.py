#!/bin/env python

"""
Convert {.gro,.top} to .pdb
"""

import os

from simtk import openmm, unit
from simtk.openmm import app

for dir, dirs, files in os.walk('.'):
    prefix = '3sy7_lig_solv_GMX'
    prefix = '3sy7_lig_nowat_GMX'
    gro_filename = prefix + '.gro'
    top_filename = prefix + '.top'
    pdb_filename = prefix + '.pdb'
    print(files)
    if (gro_filename in files) and (top_filename in files):
        grofile = app.GromacsGroFile(os.path.join(dir, gro_filename))
        topfile = app.GromacsTopFile(os.path.join(dir, top_filename), periodicBoxVectors=grofile.getPeriodicBoxVectors())
        with open(os.path.join(dir, pdb_filename), 'w') as outfile:
            app.PDBFile.writeFile(topfile.topology, grofile.positions, outfile)
