from simulation import Simulation
import numpy as np

sin, cos = np.sin, np.cos
L = [4, 3, 2]
linkparam = [(0, 0, None, 0),
             (0, 4, None, 0),
             (0, 3, None, 0),
             (0, 2, None, 0)]


def customJ(th):
    th01  = th[0] + th[1]
    th012 = th[0] + th[1] + th[2]
    j = np.array([
        [-L[0] * sin(th[0]), -L[1] * sin(th01), -L[2] * sin(th012)],
        [ L[0] * cos(th[0]),  L[1] * cos(th01),  L[2] * cos(th012)],
        [0, 0, 1]])
    j[:, 1] += j[:, 2]
    j[:, 0] += j[:, 1]
    return j


# init
sim = Simulation(linkparam, 8)
th = np.array([10, 20, 30]) / 180 * np.pi
dxy = np.array([.2, -.3, -.2])
interval = 0.1

# main
for i in np.arange(0, 5.01, interval):
    j = customJ(th)
    invj = np.linalg.inv(j)
    dth = invj.dot(dxy)
    th = th + dth * interval
    sim.sim(np.array([*(th / np.pi * 180), 0]))

sim.runAnimation(repeat=True)
