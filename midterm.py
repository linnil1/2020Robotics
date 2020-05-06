import numpy as np
from hw2 import Transform, Translation, Rotation, Base
from hw3 import Quaternion, link, symLink
import matplotlib.pyplot as plt
from simulation import Simulation
import sympy
from pprint import pprint
from gradient import gradientVariable, forwardVariable


# exit()
print("Q1")
t01 = Transform(rot=Rotation(180, 0), loc=Translation(-8, 6, 2))
tan68 = np.arctan(6 / 8) * 180 / np.pi
t13 = Transform(rot=Rotation(90 + tan68, 0) * Rotation(-90, 1),
                loc=Translation(0, 6, -8))
t03 = t01 * t13
print("T01", t01)
print("T13", t13)
print("T03", t03)
t31 = t13.T()
print("T31", t31)


q = Quaternion.fromRotationMat(t31.rot)
print("Quaternion", q)
angle, raxis = q.getRotationParam()
print("Rotation axis", raxis)
print("Rotation angle", angle)


print("Q2")
t1, t2, d3, t4, t5, t6, l1 = sympy.symbols("t1, t2, d3, t4, t5, t6, l1")
linkparam = [(  0, 0, t1, 0),
             (-90, 0, t2, l1),
             ( 90, 0,  0, d3),
             (  0, 0, t4, 0),
             (-90, 0, t5, 0),
             ( 90, 0, t6, 0)]

T = sympy.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])
for p in linkparam:
    T = T * symLink(*p)
pprint(T)


print("Q3")
tan68 = np.arctan(6 / 8) * 180 / np.pi
t_fp = Transform(rot=Rotation(90 - tan68, 1)) * \
       Transform(loc=Translation(50, 35, 0))
print("t_fp", t_fp)
t_cf = Transform(rot=Rotation(90, 0), loc=Translation(0, 0, 65))
print("t_cf", t_cf)
t_bc = Transform(rot=Rotation(90, 0), loc=Translation(0, 100, 100))
t_bp = t_bc * t_cf * t_fp
print("t_bp", t_bp)

linkparam = [(90,   0, 90,   0),
             ( 0,  20, None, 0),
             ( 0, 100, None, 0),
             ( 0, 100, None, 0),
             ( 0,  60,   90, 0),
             (90,   0,    0, 0)]

init_angle = np.array([44, 283, 264]) / 180 * np.pi
th = gradientVariable(t_bp.mat, linkparam, init_angle)
t = forwardVariable(th, linkparam)
print("Angle", th)
print("Real", t_bp)
print("Calculated", t)

if True:
    # test by simulation
    sim = Simulation(linkparam, 200)
    angle = [90, 0, 0, 0, 90, 0]
    sim.sim(angle)
    angle[1:4] = th
    sim.sim(angle)
    sim.runAnimation(repeat=False)


print("Q4")
r, a, d = sympy.symbols("r, a, d")
pi = sympy.pi
linkparam = [( 0, 0, r, 0),
             ( 0, a, 0, 0),
             (-90, 0, 0, d)]

T01 = symLink(*linkparam[0])
T12 = symLink(*linkparam[1]) * symLink(*linkparam[2])
T02 = T01 * T12
print("T01")
pprint(T01)
print("T12")
pprint(T12)
print("T02")
pprint(T02)
