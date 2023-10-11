#!/usr/bin/env python

"""
This script uses the qsm_forward library to generate BIDS-compliant files 
from a simulated MRI data. 

The simulation uses spherical susceptibility sources and different relative B0 directions
to simulate oblique acquisition.

The simulation results are saved in the "bids" directory.

Author: Ashley Stewart (a.stewart.au@gmail.com)
"""

import qsm_forward
import numpy as np

def rotate_vector(vector, axis, angle_degree):
    theta = np.radians(angle_degree)

    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    return np.dot(rotation_matrix, vector)

def generate_spherical_sources():
    # Create an empty 3D array filled with near-zeros
    shape = (150, 150, 150)
    chi = np.zeros(shape) + 0.05

    # Define the center and radius of the sphere
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    radius = 3

    # Loop through the 3D array and fill in the sphere
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if dist <= radius:
                    chi[x, y, z] = 1.0
                    chi[x, y, z+15] = 1.0
    
    return chi

if __name__ == "__main__":
    for (acq, B0_dir) in [
        (None, np.array([0, 0, 1])),
        ('rotx1', rotate_vector(np.array([0, 0, 1]), 'x', 25)),
        ('rotx2', rotate_vector(np.array([0, 0, 1]), 'x', 40)),
        ('roty1', rotate_vector(np.array([0, 0, 1]), 'y', 25)),
        ('roty2', rotate_vector(np.array([0, 0, 1]), 'y', 40)),
        ('rotxy', rotate_vector(rotate_vector(np.array([0, 0, 1]), 'x', 15), 'y', 15))
    ]:

        recon_params = qsm_forward.ReconParams(subject='spherical', B0_dir=B0_dir, acq=acq, suffix='MEGRE')

        tissue_params = qsm_forward.TissueParams(
            chi=generate_spherical_sources()
        )

        qsm_forward.generate_bids(tissue_params, recon_params, "bids")

