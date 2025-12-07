#!/usr/bin/env python3

import numpy as np
import scipy as sp

import eq, ops

class NN:
    def __init__(self, N):
        self.N = N

        self.z = ops.circular_gaussian(self.N, self.N//2, 20.0, 0.375)
        self.z0 = 0.69
        self.X = np.zeros(N)
        self.dW = np.zeros(N)

        self.A = 45/N
        self.wij = self.A * eq.gaussian(ops.dist_matrix(N, 'rad'), 1.5)

        self.gammaE = -1.5
        self.gammaI = -7.5

        self.tauE = 0.005
        self.tauI = 0.00025

        self.t = 0.0
        self.dt = 0.0001

    def step(self, q, u, c, v, delta, sigma, n_opt, theta, offset, shape):
        mu0 = [v] + [v * delta] * (int(n_opt) - 1)
        self.X, _ = ops.create_N_targets(self.N, mu0, shape, sigma, theta=theta, offset=offset, bypass=False)

        self.dW = c * np.sqrt(self.dt) * np.random.normal(0, 1, size=self.N) * self.z

        def rhs(t, x):
            z = x[:-1]
            z0 = x[-1]

            dz_dt = eq.ring_neuron(t, z, z0, self.wij, q, self.gammaE, self.tauE, self.X, self.dW)
            dz0_dt = eq.pooled_inhibition(t, z, z0, u, self.gammaI, self.tauI)
            return np.concatenate([dz_dt, [dz0_dt]])

        z_state = np.concatenate([self.z, [self.z0]])
        sol = sp.integrate.solve_ivp(rhs,
                                     (self.t, self.t + self.dt), 
                                     z_state, 
                                     method="RK45")

        z_next = sol.y[:, -1]

        self.z = z_next[:-1]
        self.z0 = z_next[-1]

        self.t += self.dt