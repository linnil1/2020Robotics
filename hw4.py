import numpy as np
from hw2 import Transform, Translation, Rotation, Base
from hw3 import Quaternion
import matplotlib.pyplot as plt
from pprint import pprint

sin, cos = np.sin, np.cos


def vTranslate(v=Translation(), w=Translation(), mat=None):
    if mat is not None:
        return Translation(mat=mat)
    return Translation(mat=np.hstack([v.mat, w.mat]))


def vTransform(rot, p_rel=Translation()):
    p = p_rel.mat
    if type(rot) is not Rotation:
        raise ValueError("Wrong Rotation")
    s = [[    0, -p[2],  p[1]],
         [ p[2],     0, -p[0]],
         [-p[1],  p[0],    0]]
    mat = np.vstack([
        np.hstack([rot.mat, -rot.mat.dot(s)]),
        np.hstack([np.zeros([3, 3]), rot.mat])])
    return Transform(mat=mat)


if __name__ == "__main__":
    print("HW3 Q1")
    r = Rotation(30, 2).T()
    t = Translation(10, 0, 5)

    v = Translation(0, 2, -3)
    w = Translation(1.414, 1.414, 0)

    T = vTransform(r, t)
    p = vTranslate(v, w)
    print(T * p)


    print("HW4 Q1")
    def getrotth(th1, dth):
        r1 = Rotation(th1[0], 2) * Rotation(th1[1], 1) * Rotation(th1[2], 2)
        th2 = th1 + dth
        r2 = Rotation(th2[0], 2) * Rotation(th2[1], 1) * Rotation(th2[2], 2)
        t1 = th1 / 180 * np.pi
        t2 = dth / 180 * np.pi
        # print(r1.getXYZ())
        # print(r2.getXYZ())
        dxyz = np.array(r2.getXYZ()) - np.array(r1.getXYZ())
        return t1, t2, dxyz / 180 * np.pi

    # test
    th1 = np.array([-130, 80, 10])
    dth = np.array([2, -1, 1])
    th1 = np.array([130, -20, 100])
    dth = np.array([2, -1, 1])
    th1 = np.array([100, 200, -100])
    dth = np.array([-2, 1, 2])
    th1 = np.array([100, 200, -100])
    dth = np.array([-20, 10, 20])

    # run here
    th1 = np.array([10, 200, -10])
    dth = np.array([1, 0, 2])
    t1, t2, dxyz = getrotth(th1, dth)
    w1 = -t2[0] * sin(t1[1]) * cos(t1[2]) + t2[1] * sin(t1[2])
    w2 =  t2[0] * sin(t1[1]) * sin(t1[2]) + t2[1] * cos(t1[2])
    w3 =  t2[0] * cos(t1[1])              + t2[2]
    print(dxyz)
    print(w3, w2, w1)


    print("HW4 Q2")
    L = [4, 3, 2]
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
    th = np.array([10, 20, 30]) / 180 * np.pi
    xy = np.array([0, 0, 0])  # assume
    fw = np.array([1, 2, 3])
    dxy = np.array([.2, -.3, -.2])

    # i, theta, dtheta, position, det_j, torque, Real_XY
    arr = []
    interval = 0.1
    for i in np.arange(0, 5.01, interval):
        j = customJ(th)
        invj = np.linalg.inv(j)
        dth = invj.dot(dxy)
        th = th + dth * interval
        xy = xy + dxy * interval
        torq = invj.dot(fw)
        thd = th / np.pi * 180
        t = Transform(rot=Rotation(thd[0], 2)) * \
            Transform(rot=Rotation(thd[1], 2), loc=Translation(L[0], 0, 0)) * \
            Transform(rot=Rotation(thd[2], 2), loc=Translation(L[1], 0, 0)) * \
            Transform(                         loc=Translation(L[2], 0, 0))
        tmp = Translation(1, 0, 0)
        arr.append((i, *th, *dth, *xy, np.linalg.det(j), *torq,
                    *t.loc.mat[:2], *(t.T() * tmp).mat[:2]))

    arr = np.array(arr)
    print(arr)

    # plot it
    plt.title("Joint Angle")
    plt.plot(arr[:, 0], arr[:, 1:4])
    plt.xlabel("Time(s)")
    plt.ylabel("(rad)")
    plt.legend(["1", "2", "3"])
    plt.savefig("hw4_2_0.png")
    plt.clf()

    plt.title("Joint angular velcoity")
    plt.plot(arr[:, 0], arr[:, 4:7])
    plt.xlabel("Time(s)")
    plt.ylabel("(rad/s)")
    plt.legend(["1", "2", "3"])
    plt.savefig("hw4_2_1.png")
    plt.clf()

    plt.title("XY component")
    plt.plot(arr[:, 0], arr[:, 7:10])
    plt.xlabel("Time(s)")
    plt.ylabel("(m)")
    plt.legend(["1", "2", "Direction(rad)"])
    plt.savefig("hw4_2_2.png")
    plt.clf()

    plt.title("Determinant")
    plt.plot(arr[:, 0], arr[:, 10])
    plt.xlabel("Time(s)")
    plt.savefig("hw4_2_3.png")
    plt.clf()

    plt.title("Torque")
    plt.plot(arr[:, 0], arr[:, 11:14])
    plt.xlabel("Time(s)")
    plt.ylabel("(N*m)")
    plt.legend(["1", "2", "3(N*m^2)"])
    plt.savefig("hw4_2_4.png")
    plt.clf()

    plt.title("Plot on XY")
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    plt.plot(arr[:, 14], arr[:, 15], 'o')
    for i in range(len(arr)):
        plt.text(arr[i, 14], arr[i, 15], str(round(arr[i, 0], 1)))
        plt.gca().arrow(arr[i, 14], arr[i, 15],
                        arr[i, 16] / 100, arr[i, 17] / 100)
    plt.savefig("hw4_2_5.png")
    plt.show()
    plt.clf()
