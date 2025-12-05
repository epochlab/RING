#!/usr/bin/env python3

import numpy as np

def relu(x): return np.maximum(0, x)

def ring_neuron(t, z, z0, wij, q, gammaE, tauE, X, dW):
    R = relu(gammaE + np.dot(wij, z) - q * z0 + X + dW)
    return (-z + R) / tauE

def pooled_inhibition(t, z, z0, u, gammaI, tauI):
    R = relu(gammaI + u * np.sum(z) - z0)
    return (-z0 + R) / tauI