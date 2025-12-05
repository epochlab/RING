#!/usr/bin/env python3

import numpy as np

def n2deg(n, N): return (n / N) * 360
def deg2n(deg, N): return (deg / 360) * N

def distance(x, y): return np.sum((x - y)**2, axis=1)

def dist_matrix(N, output=None):
    arr = np.arange(N)                              # Array length N
    dij = np.abs(arr[:, None] - arr)                # N by N grid of units
    dij = np.minimum(dij, N - dij)                  # Mirror / wrapping
    nij = dij / N                                   # Normalise

    if output=='rad': return nij * 2 * np.pi        # Return radians (0-π :: max wrap is N/2)
    return nij

def construct_shape(N, mean, sigma, A, shape="gaussian"):
    dist = np.abs(np.arange(N) - mean)
    dist = np.minimum(dist, N - dist)
    if shape == "gaussian": return A * np.exp(-0.5 * (dist / sigma)**2)
    if shape == "triangle": return A * np.maximum(0, 1 - dist / sigma)
    if shape == "laplace": return A * np.exp(-dist / sigma)

def circular_gaussian(N, mean, sigma, A):
    dist = np.abs(np.arange(N) - mean)
    dist = np.minimum(dist, N - dist)
    return np.exp(-0.5 * (dist / sigma)**2) * A

def create_N_targets(N, A, shape, sigma, theta=None, offset=0, bypass=False):
    num_targets = len(A)
    offset = int(round(offset % N))
    center = (N // 2 + offset) % N

    sep = theta / num_targets
    start = center - theta//2
    idx = (start + (np.arange(num_targets) + 0.5) * sep) % N

    x = np.zeros(N)
    for id, a in zip(idx, A):
        if bypass: x[id.astype(int)] = a
        else:
            # x += circular_gaussian(N, id, sigma, a)
            x += construct_shape(N, id, sigma, a, shape=shape)
    return x, idx

def com_trajectory(z, N):
    angles_rad = 2 * np.pi * np.arange(N) / N
    cos_sum = z @ np.cos(angles_rad)
    sin_sum = z @ np.sin(angles_rad)
    mean_angle_rad = np.arctan2(sin_sum, cos_sum)
    return (np.degrees(mean_angle_rad) + 360.0) % 360.0