import numpy as np
from simulation import Simulation
from gradient import gradientVariable, forwardVariable
from utils import Transform, Translation, Rotation, ArmDynamic
import matplotlib.pyplot as plt

# DH
linkparam = [[90,   0, 90,   0],
             [ 0, 0.02, None, 0],
             [ 0, 0.05, None, 0],
             [ 0, 0.08, None, 0],
             [ 0, 0.04,   90, 0],
             [90,   0,    0, 0]]
g = 9.8
M = [0, 0.2, 0.2, 0.2, 0, 0]
I = [0, 0, 0, [[0, 0, 0], [0, 0, 0], [0, 0, 0.01]], 0, 0]
C = [[0, 0, 0], [0.01, 0, 0], [0.025, 0, 0], [0.04, 0, 0], [0.02, 0, 0], [0, 0, 0]]

# init position
th0 = np.array([0, 0, 0])
init_angle = th0 / 180 * np.pi

# find target position
tan68 = np.arctan(6 / 8) * 180 / np.pi
t_fp = Transform(rot=Rotation(90 - tan68, 1)) * \
       Transform(loc=Translation(0.050, 0.035, 0))
t_cf = Transform(rot=Rotation(90, 0), loc=Translation(0, 0, 0.065))
t_bc = Transform(rot=Rotation(90, 0), loc=Translation(0, 0.100, 0.100))
t_bp = t_bc * t_cf * t_fp
print("Target", t_bp)
th1 = gradientVariable(t_bp.mat, linkparam, init_angle)

# set rotation angle that < 180 degrees 
th1[(th1 - th0) >  180] -= 360
th1[(th1 - th0) < -180] += 360
vth = [ 0, *(th1 - th0) / 10, 0, 0]
ath = [ 0,  0,  0,  0,  0, 0]
print("Angular velcoity", vth)

# init
torque = []
sim = Simulation(linkparam, 0.1)
arms = ArmDynamic(linkparam)
arms.addGravity(2, g)  # on z
arms.M = M
arms.I = I
arms.C = C

# run
ths = np.linspace(th0, th1)
for angle in ths:
    th = [90, *angle, 90, 0]
    sim.sim(th)
    arms.clean()
    arms.run(th, vth, ath)
    torque.append([i.mat[2] for i in arms.n])

# plot
sim.runAnimation(repeat=True)
# save
"""
# ani = sim.runAnimation(show=False)
# ani.save('final_sim0.gif', dpi=80, writer='imagemagick')
"""

# show needed torque for motor
print("Needed Torque")
print(torque)
torque = np.array(torque)
fig = plt.figure()
ax = fig.gca()
ax.plot(np.arange(torque.shape[0]), np.abs(torque))
# save
# plt.savefig("final_sim0.png")
plt.show()
