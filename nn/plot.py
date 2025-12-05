#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import ops

def simple_xy(x, y, title, x_label, y_label):
    plt.plot(x, y, 'k', lw=0.5)
    plt.title(title)
    plt.xlabel(x_label), plt.ylabel(y_label)
    plt.xlim([x[0], x[-1]])
    plt.grid(alpha=0.25)

def matrix(x, title, x_label, y_label):
    plt.imshow(x)
    plt.title(title)
    plt.xlabel(x_label), plt.ylabel(y_label)

def neural_activity(N, n_steps, X, z, dW, y_min=-0.1, y_max=1.2):
    plt.axhline(1.0, lw=0.5, color='gray')

    if X is not None: plt.plot(X, 'r--', lw=0.5, label="Input projection to ring (Xₑ)")
    if z is not None: plt.plot(z, 'o-', c='k', lw=1, label="Ring attractor activity (zₑ)", markerfacecolor='w', markersize=4)
    if dW is not None: plt.plot(dW, lw=0.5, label="Noise: Ring (dWr)")

    plt.title(f'Ring Attractor Dynamics | N={N} | Timesteps = {n_steps}')
    plt.xlabel(r"$\phi$"), plt.ylabel("Response")
    plt.ylim([y_min, y_max])
    plt.legend(loc='upper right'), plt.grid(alpha=0.25)

def time_evolution(N, time_ms, z_hist, n_samples=6):
    activity_subset = np.vstack(z_hist)
    activity_subset[0, 0] = 0
    activity_subset[0, -1] = 0
    sample_idx = np.linspace(0, activity_subset.shape[0]-1, n_samples, dtype=int)

    for i in sample_idx:
        pX = np.arange(N)
        pY = np.ones_like(pX) * time_ms[i]
        pZ = activity_subset[i]

        verts = [list(zip(pX, pY, pZ))]
        plt.gca().add_collection3d(Poly3DCollection([verts[0]], facecolors='white', edgecolors='k', lw=0.25, alpha=0.75))

    plt.xlabel('Neural Population ($N$)')
    plt.ylabel('Time (ms)')
    plt.gca().set_zlabel('Response')
    plt.gca().set_ylim(time_ms[0], time_ms[-1])
    plt.gca().view_init(35, -60)
    plt.gca().set_box_aspect((9, 12, 4))

def weight_profile(N, wij, ic):    
    plt.plot(wij[:, N//2], lw=0.5, label='Weight Profile')
    plt.plot(ic, lw=0.5, label='Initial Condition')
    plt.title("Weight Profile & Initial Condition")
    plt.xlabel(r"$z_i$"), plt.ylabel("Weight")
    plt.xlim(0, N - 1)
    plt.grid(alpha=0.25), plt.legend()

def com_trajectory(N, t, z_hist, idx, mu0):
    com_deg = ops.com_trajectory(z_hist, N)
    com_angles = np.deg2rad(com_deg + 180)
    target = np.deg2rad((idx / N) * 360 + 180)
    r_max = t.max() if t.size else 1
    s = 10 + 200 * np.array(mu0)

    if len(target) >= 2:
        a1, a2 = target[:2]
        d = abs(a2 - a1)
        if d > np.pi:
            d = 2 * np.pi - d
            if a2 < a1:
                a1, a2 = a2, a1
        arc = np.linspace(a1, a2, 100)
        plt.plot(arc, np.full_like(arc, r_max), "b--", lw=1)
        plt.text((a1 + a2)/2, r_max*0.8, f"{np.rad2deg(d):.1f}°",
                ha="center", va="center", fontsize=10, color="blue")

    plt.scatter(target, np.full_like(target, r_max), s=s, facecolors='lightgreen', edgecolors='k')
    plt.plot(com_angles, t, "r", lw=1)
    plt.plot(com_angles[-1], t[-1], "ro", ms=4)

    plt.title("Centre of Mass Trajectory")
    plt.gca().set_theta_zero_location('N')
    plt.gca().set_theta_direction(-1)
    plt.gca().set_rlabel_position(0)