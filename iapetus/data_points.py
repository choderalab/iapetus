"""

This module extracts the coordinates to be used in the cylinder fitting.

"""
import sys
import numpy as np
import mdtraj as md
import parmed as pmd
from mdtraj import geometry

class DataPoints(object):

    def __init__(self, structure, topology):

        self.structure = structure
        self.indices = topology.select('protein')
        self.sliced_top = topology.subset(self.indices)
        self.coordinates = self._points()

    def _points(self):

        coords = self.structure.get_coordinates(frame=0)[self.indices,:]
        xyz = np.zeros(shape=(1,coords.shape[0],3))
        xyz[0,:,:] = coords/10.0 # Coords in nanometers for mdTraj
        traj = md.Trajectory(xyz, topology = self.sliced_top)
        secondary = geometry.compute_dssp(traj, simplified=False)
        strands = list(filter(lambda i: secondary[0,i] == 'E',range(secondary.shape[1])))
        coordinates = []
        for s in strands:
            for res in self.structure.residues:
                if res.idx == s:
                    for item in res.atoms:
                        if item.name == 'CA':
                            coordinates.append(coords[item.idx,:])

        return np.asarray(coordinates)

    def writeCoordinates(self, file=sys.stdout):
        print("{}\n".format(self.coordinates.shape[0]), file=file)
        for row in self.coordinates:
            print("C {} {} {}".format(row[0], row[1], row[2]), file=file)
