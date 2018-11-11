"""

This module fits a cylinder to data points.

Search for two sets of n data points.  The geometrical
center of each set determine the bottom and top coordinates.

"""
import sys
import random
import itertools
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import least_squares

class CylinderFitting(object):
    """
    Properties
    -----------

    """
    def __init__(self, xyz):

        cov_matrix = np.cov(np.asarray(xyz).transpose())
        w, v = np.linalg.eig(cov_matrix)
        deviation =[[] for i in range(w.shape[0])]
        dev = np.Inf
        for e in range(w.shape[0]):
            eigen = v[:,np.where(w == w[e])]
            eigen.resize(3,1)
            phi_ = np.arcsin(eigen[2,0])
            theta_ = np.arcsin(np.cos(phi_))
            self.center = center = xyz.mean(axis=0)
            self.W = np.array([np.cos(theta_)*np.cos(phi_),
                            np.sin(theta_)*np.cos(phi_),
                            np.sin(phi_)]) # W is the cylinder axis
            R = ((np.linalg.norm((self._project(xyz)),axis=1)).mean())**2
            for i in range(xyz.shape[0]):
                deviation[e].append((np.linalg.norm(self._project(xyz)[i,:])**2 - R))
            if sum(deviation[e]) < dev:
                dev = sum(deviation[e])
                phi = phi_
                theta = theta_

        self.W = np.array([np.cos(theta)*np.cos(phi),
                        np.sin(theta)*np.cos(phi),
                        np.sin(phi)])

        params = [self.center[0], self.center[1], theta, phi,
                 ((np.linalg.norm((self._project(xyz)),axis=1)).mean())**2] # params[4] = radius

        estParams = least_squares(self._cylinderFitting, params, args=(xyz,), f_scale=0.1)
        self.center[0:2] = estParams.x[0:2]
        theta, phi = tuple(estParams.x[2:4])

        self.r = np.sqrt(estParams.x[4])
        self.W = np.array([np.cos(theta)*np.cos(phi),np.sin(theta)*np.cos(phi),np.sin(phi)])
        self.bottom, self.top, self.height = self._computeExtremes(xyz)

    def _cylinderFitting(self, params, xyz):

        """
        Reference:

        params are variables used for computing the error function in the
        fitting procedure

        params[0] = x coordinate of the cylinder centre
        params[1] = y coordinate of the cylinder centre
        params[2] = theta, rotation angle about the z-axis
        params[3] = phi, orientation angle of the plane with normal vector W
        params[4] = r, radius of the cylinder

        xyz are the points to fit

        """
        x, y, theta, phi, R = tuple(params)
        z = xyz[:,2].mean()
        self.center = np.array([x, y, z])
        self.W = np.array([np.cos(theta)*np.cos(phi),
                      np.sin(theta)*np.cos(phi),
                      np.sin(phi)])

        deviation = []

        for i in range(xyz.shape[0]):
            deviation.append((np.linalg.norm(self._project(xyz)[i,:])**2 - R))

        return deviation

    def _project(self,xyz):

        plane = np.identity(3) - np.dot(self.W[:,np.newaxis],self.W[np.newaxis,:])
        projection = []
        for data in range(xyz.shape[0]):
            vector = xyz[data,:] - self.center
            projection.append(np.dot(vector,plane))
        return np.asarray(projection)

    def _computeExtremes(self,xyz):

        heights = []
        for data in range(xyz.shape[0]):
            heights.append(np.inner((xyz[data,:]-self.center),self.W))
        hpoints = np.asarray(heights)
        bottom, top = self.center + min(hpoints)*self.W, self.center + max(hpoints)*self.W
        height = max(hpoints) - min(hpoints)
        self.center = (top + bottom)/2 # Recalculate the cylinder center
        bottom, top = self.center - (height/2)*self.W, self.center + (height/2)*self.W

        return bottom, top, height

    def _axisCylinder(self,xyz):

        p = self._project(xyz).tolist()
        V = random.sample(p, 1)/np.linalg.norm(random.sample(p, 1))
        U = np.cross(V,self.W)

        return U, V

    def vmdCommands(self):

        commands  = "set bottom {{ {} {} {} }}\n".format(*self.bottom)
        commands += "set top {{ {} {} {} }}\n".format(*self.top)
        commands += "draw material Transparent\n"
        commands += "draw color silver\n"
        commands += "draw cylinder $bottom $top radius {} resolution 100\n".format(self.r)

        return commands

    def atomsInExtremes(self, xyz, n=5, weights=None):

        """

        Finds two combinations of points whose (possibly weighted) mean coordinates are
        the closest to the cylinder extremities. The points are split into a number of sectors
        according to their azimuth angles and the returned combinations will contain one particle
        of each sector.

        """
        nsectors = 2*n
        delta = 2*np.pi/nsectors
        sector = [[] for i in range(nsectors)]
        bottom_sector = [[] for i in range(nsectors)]
        top_sector = [[] for i in range(nsectors)]
        U, V = self._axisCylinder(xyz)
        for i in range(xyz.shape[0]):
            xyz_u = np.inner((xyz[i,:]- self.center), U)
            xyz_v = np.inner((xyz[i,:]- self.center), V)
            angle = np.arctan2(xyz_u, xyz_v)
            isec = int((angle + np.pi)/delta)
            height = np.inner((xyz[i,:]-self.center),self.W)
            if height < 0:
                bottom_sector[isec].append(i)
            else:
                top_sector[isec].append(i)

        d0_b = d0_t = np.Inf
        for comb in itertools.product(*itertools.islice(bottom_sector, 0, None, 2)):
            masses = None if weights is None else weights[list(comb)]
            coords = np.average(xyz[list(comb),:], axis=0, weights=masses)
            d_bottom = np.linalg.norm(coords - self.bottom)
            if d_bottom < d0_b:
                d0_b = d_bottom
                bottom_atoms = comb

        for comb in itertools.product(*itertools.islice(bottom_sector, 1, None, 2)):
            masses = None if weights is None else weights[list(comb)]
            coords = np.average(xyz[list(comb),:], axis=0, weights=masses)
            d_bottom = np.linalg.norm(coords - self.bottom)
            if d_bottom < d0_b:
                d0_b = d_bottom
                bottom_atoms = comb


        for comb in itertools.product(*itertools.islice(top_sector, 0, None, 2)):
            masses = None if weights is None else weights[list(comb)]
            coords = np.average(xyz[list(comb),:], axis=0, weights=masses)
            d_top = np.linalg.norm(coords - self.top)
            if d_top < d0_t:
                d0_t = d_top
                top_atoms = comb

        for comb in itertools.product(*itertools.islice(top_sector, 1, None, 2)):
            masses = None if weights is None else weights[list(comb)]
            coords = np.average(xyz[list(comb),:], axis=0, weights=masses)
            d_top = np.linalg.norm(coords - self.top)
            if d_top < d0_t:
                d0_t = d_top
                top_atoms = comb

        return bottom_atoms, top_atoms

    def writeExtremesCoords(self, xyz, bottom_atoms, top_atoms, file=sys.stdout):

        bottom_center = xyz[[int(i) for i in bottom_atoms], :].mean(0)
        top_center = xyz[[int(i) for i in top_atoms], :].mean(0)
        data = 2*len(bottom_atoms) + 4
        print("{}\n".format(data),file=file)
        print("O {} {} {}".format(*self.bottom),file=file) # cylinder center
        print("O {} {} {}".format(*self.top),file=file) # # cylinder center
        print("C {} {} {}".format(*bottom_center),file=file)
        print("C {} {} {}".format(*top_center),file=file)
        for i in bottom_atoms + top_atoms:
            print("C {} {} {}".format(*xyz[i]),file=file)
