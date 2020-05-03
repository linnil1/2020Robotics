import numpy as np
from hw2 import Transform, Translation, Rotation, Base
from hw3 import Quaternion, link
from hw4 import vTransform, vTranslate
import matplotlib.pyplot as plt
from pprint import pprint

sin, cos = np.sin, np.cos

# The arm setting
"""
check = True
linkparam = [[ 0,   0, None, 0],
             [ 0,  30, None, 0],
             [ 0,  20,    0, 0]]
g = 9.8
th = [20, 30, 0]
vth = [-10, 15, 0]
ath = [15, -20, 0]
M = [0, 0.1, 0.2]
I = [0, 0, 0]
C = [[0, 0, 0], [30, 0, 0], [20, 0, 0]]

check = True
linkparam = [[ 0,   0, None, 0],
             [ 0,  3.0, None, 0],
             [ 0,  4.0,    0, 0]]
g = 9.8
th = [20, -10, 0]
vth = [10, -15, 0]
ath = [-12, 20, 0]
M = [0, 0.3, 0.1]
I = [0, 0, 0]
C = [[0, 0, 0], [3, 0, 0], [4, 0, 0]]

"""
check = False
linkparam = [[ 0,    0, None, 0],
             [ 0,  0.5, None, 0],
             [ 0,  0.5, None, 0],
             [ 0,    1,    0, 0]]
g = 9.8
th  = [10, 20, 30, 0]
vth = [1, 2, 3, 0]
ath = [0.5, 1, 1.5, 0]
M = [0, 4.6, 2.3, 1]
I = [0, 0, 0, [[0.5, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]]
C = [[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0, 0, 0]]

# Result
Trans = [Transform()]
v = [vTranslate()]
a = [vTranslate(v=Translation(0, g, 0))]
xy = [Translation()]
ac = [Translation()]
F  = [Translation()]
N  = [Translation()]
f  = [Translation()] * (len(linkparam) + 1)
n  = [Translation()] * (len(linkparam) + 1)

# joint
for i in range(len(linkparam)):
    # set angle of joint
    linkparam[i][2] = th[i]

    # transform matrix for angle
    T = link(*linkparam[i])
    Trans.append(Trans[-1] * T)

    # Space location i+1
    xy.append(Trans[-1] * Translation())

    # transform matrix for velcoity
    T_v = vTransform(Rotation(mat=T.rot.mat.T), T.loc)

    # velcoity i+1 (R-joint)
    new_v = T_v * v[i]
    new_v.mat[5] += vth[i] / 180 * np.pi
    v.append(new_v)

    # acceleration i+1 (R-joint)
    new_a = T_v * a[i]
    w = v[i].mat[3:6]
    new_a.mat[5] += ath[i] / 180 * np.pi
    new_a.mat[0:3] += T.rot.mat.T.dot(np.cross(w, np.cross(w, T.loc.mat)))
    new_a.mat[3:6] += T.rot.mat.T.dot(np.cross(w, [0, 0, vth[i] / 180 * np.pi]))
    a.append(new_a)


# link
for i in range(1, len(linkparam)):
    # acceleration i+1 on center(Center located at i+1)
    center = C[i]
    new_w = v[i].mat[3:6]
    new_a = a[i]
    new_ac = new_a.mat[0:3] + np.cross(new_w, np.cross(new_w, center)) + np.cross(new_a.mat[3:6], center)
    ac.append(new_ac)

    # force i+1 on center
    F.append(Translation(mat=new_ac * M[i]))

    # Torque i+1 on center
    now_I = np.array(I[i])
    N.append(Translation(mat=now_I.dot(new_a.mat[3:6]) + \
                             np.cross(new_w, now_I.dot(new_w))))


for i in reversed(range(len(linkparam))):
    # force on motor
    T = Trans[i + 1]
    f[i] = F[i] + T.rot * f[i + 1]

    # torque on motor
    center = C[i]
    n[i] = N[i] + T.rot * n[i + 1] + Translation(mat=np.cross(center, F[i].mat) + \
           np.cross(T.loc.mat, (T.rot * f[i + 1]).mat))

# remove last one
f = f[:-1]
n = n[:-1]

th = np.array(th) / 180 * np.pi
vth = np.array(vth) / 180 * np.pi
ath = np.array(ath) / 180 * np.pi
print("th",  th, vth, ath, sep="\n")
print("xy", *xy, sep="\n")
print("v",   *v, sep="\n")
print("a",   *a, sep="\n")
print("ac", *ac, sep="\n")
print("F",   *F, sep="\n")
print("N",   *N, sep="\n")
print("f",   *f, sep="\n")
print("n",   *n, sep="\n")

# check
if not check:
    exit()
print("a1x", g * sin(th[0]))
print("a1y", g * cos(th[0]))
l = [linkparam[1][1], linkparam[2][1]]
print("a2x",  l[0] * ath[0] * sin(th[1]) - l[0] * vth[0] ** 2 * cos(th[1]) + g * sin(th.sum()))
print("a2y",  l[0] * ath[0] * cos(th[1]) + l[0] * vth[0] ** 2 * sin(th[1]) + g * cos(th.sum()))
print("ac1x",-l[0] * vth[0] ** 2 + g * sin(th[0]))
print("ac1y", l[0] * ath[0]      + g * cos(th[0]))
print("ac2x", l[0] * ath[0] * sin(th[1]) - l[0] * vth[0] ** 2 * cos(th[1]) + g * sin(th.sum()) - l[1] * vth.sum() ** 2)
print("ac2y", l[0] * ath[0] * cos(th[1]) + l[0] * vth[0] ** 2 * sin(th[1]) + g * cos(th.sum()) + l[1] * ath.sum())

print("n2", M[2] * l[1] ** 2 * ath.sum() + \
            M[2] * l[0] * l[1] * (cos(th[1]) * ath[0] + sin(th[1]) * vth[0] ** 2) + \
            M[2] * l[1] * g * cos(th.sum()))

print("n1", M[2] * l[1] ** 2 * ath.sum() + \
            M[2] * l[0] * l[1] * (cos(th[1]) * (2 * ath[0] + ath[1]) - sin(th[1]) * vth[1] ** 2 - 2 * sin(th[1]) * vth[0] * vth[1]) + \
            (M[1] + M[2]) * (l[0] ** 2 * ath[0] + l[0] * g * cos(th[0])) + \
            M[2] * l[1] * g * cos(th.sum())
)
