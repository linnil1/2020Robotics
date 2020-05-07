import numpy as np

np.set_printoptions(3, suppress=True)
sin, cos = np.sin, np.cos


class Base:
    def __init__(self, n=3, mat=None):
        if mat is not None:
            self.mat = np.array(mat, dtype=np.float)
            return
        self.mat = np.identity(n)

    def __str__(self):
        return f"{self.mat}"

    def norm(self):
        return np.sqrt(np.sum(self.mat ** 2))

    def __add__(self, other):
        return self.__class__(mat=self.mat + other.mat)


class Rotation(Base):
    """
    Rotation matrix.
    Default is 3x3
    """
    def __init__(self, th=0, axis=0, mat=None):
        """
        Get rotation matrix rounded around specific axis.
        axis=0 for x
        axis=1 for y
        axis=2 for z
        """
        if mat is not None:
            super().__init__(mat=mat)
            return
        r = np.identity(3)
        r[(axis + 1) % 3, (axis + 1) % 3] = \
        r[(axis + 2) % 3, (axis + 2) % 3] = np.cos(th * np.pi / 180)
        r[(axis + 1) % 3, (axis + 2) % 3] = np.sin(th * np.pi / 180) * -1
        r[(axis + 2) % 3, (axis + 1) % 3] = np.sin(th * np.pi / 180)
        self.mat = r

    def __mul__(self, other):
        return Rotation(mat=self.mat.dot(other.mat))

    def T(self):
        """ Inverse """
        return Rotation(mat=np.linalg.inv(self.mat))

    def getXYZ(self):
        """ Get a, b, r from fixed xyz or eulur zyx """
        v = self.mat
        b = np.arctan2(-v[2, 0], np.sqrt(v[0, 0] ** 2 + v[1, 0] ** 2))
        cb = np.cos(b)
        a = np.arctan2(v[1, 0] / cb, v[0, 0] / cb)
        r = np.arctan2(v[2, 1] / cb, v[2, 2] / cb)
        return a / np.pi * 180, b / np.pi * 180, r / np.pi * 180

    def getZYZ(self):
        """ Get a, b, r from Eulur zyz """
        v = self.mat
        b = np.arctan2(np.sqrt(v[2, 0] ** 2 + v[2, 1] ** 2), v[2, 2])
        sb = np.sin(b)
        a = np.arctan2(v[1, 2] / sb,  v[0, 2] / sb)
        r = np.arctan2(v[2, 1] / sb, -v[2, 0] / sb)
        return a / np.pi * 180, b / np.pi * 180, r / np.pi * 180


class Translation(Base):
    """
    Vector in 3x1
    """
    def __init__(self, x=0, y=0, z=0, mat=None):
        if mat is not None:
            super().__init__(mat=mat)
            return
        self.mat = np.array([x, y, z])

    def __rmul__(self, other):
        """ Allow mulitply with 4x4 """
        if other.mat.shape[0] != self.mat.shape[0]:
            v = np.hstack([self.mat, 1])
            return Translation(mat=other.mat.dot(v)[:3])
        return Translation(mat=other.mat.dot(self.mat))


class Transform(Base):
    """
    Transform matrix (4x4)
    """
    def __init__(self, rot=Rotation(), loc=Translation(), mat=None):
        if mat is None:
            super().__init__(4)
        else:
            super().__init__(mat=mat)
            rot = Rotation(mat=mat[:3, :3])
            loc = Translation(mat=mat[0:3, 3])

        self.rot = rot
        self.mat[0:3, 0:3] = rot.mat
        self.loc = loc
        self.mat[0:3, 3] = loc.mat.reshape(self.mat[0:3 ,0].shape)

    def T(self):
        """ Inverse """
        return Transform(mat=np.linalg.inv(self.mat))

    def __mul__(self, other):
        if other.mat.shape[0] != self.mat.shape[0]:
            return NotImplemented
        if type(other) is not Transform:
            return NotImplemented
        return Transform(mat=self.mat.dot(other.mat))


class Quaternion(Base):
    """ Quaternion """
    def __init__(self, c=0, i=0, j=0, k=0, mat=None):
        if mat is not None:
            super().__init__(mat=mat)
            return
        self.mat = np.array([c, i, j, k])

    def __mul__(self, other):
        if type(other) is not Quaternion:
            return NotImplemented
        return Quaternion(mat=self.getRotationMat().dot(other.mat))

    def T(self):
        """ Inverse """
        return Quaternion(self.mat[0], *(-self.mat[1:]))

    def getRotationMat(self):
        """ Output as matrix """
        q = self.mat / self.norm()
        return np.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1],  q[0], -q[3],  q[2]],
            [q[2],  q[3],  q[0], -q[1]],
            [q[3], -q[2],  q[1],  q[0]]])

    @classmethod
    def fromRotationMat(cls, rot):
        """ Input from matrix """
        r = rot.mat
        angle = np.arccos((np.trace(r) - 1) / 2)
        raxis = 1 / 2 / np.sin(angle) * np.array([
            r[2,1] - r[1,2],
            r[0,2] - r[2,0],
            r[1,0] - r[0,1]])
        return cls.setRotation(angle / np.pi * 180, raxis)

    def getRotationParam(self):
        """ Rotation based on [x, y, z] with theta """
        q = self.mat / self.norm()
        th = np.arccos(q[0])
        if th < 1e-3:
            return 0, [0, 0, 0]
        # 0 < th < pi
        return th / np.pi * 180 * 2, q[1:] / np.sin(th)

    @classmethod
    def setRotation(cls, th, loc):
        """ Rotation based on [x, y, z] with theta """
        th = th / 180 * np.pi / 2
        return Quaternion(np.cos(th),
                          *(np.sin(th) * np.array(loc)))

    def getIJK(self):
        """ Get i,j,k value """
        return self.mat[1:]


def link(twist, dist, angle, offset):
    """
    Transform matrix of this link with DH parameters.
    """
    T1 = Transform(rot=Rotation(twist, 0))
    T2 = Transform(loc=Translation(dist, 0, 0))
    T3 = Transform(rot=Rotation(angle, 2))
    T4 = Transform(loc=Translation(0, 0, offset))
    return T1 * T2 * T3 * T4


def symLink(twist, dist, angle, offset):
    """
    Transform matrix of this link with DH parameters.
    (Use symbols)
    """
    twist = twist * sympy.pi / 180
    T1 = sympy.Matrix([
        [1, 0, 0, dist],
        [0, sympy.cos(twist), -sympy.sin(twist), 0],
        [0, sympy.sin(twist),  sympy.cos(twist), 0],
        [0, 0, 0, 1]])
    # T1[sympy.abs(T1) < 1e-3] = 0
    T2 = sympy.Matrix([
        [sympy.cos(angle), -sympy.sin(angle), 0, 0],
        [sympy.sin(angle),  sympy.cos(angle), 0, 0],
        [0, 0, 1, offset],
        [0, 0, 0, 1]])
    return T1 * T2


def vTranslate(v=Translation(), w=Translation(), mat=None):
    """ Vector of velcoity and angular (6x1)"""
    if mat is not None:
        return Translation(mat=mat)
    return Translation(mat=np.hstack([v.mat, w.mat]))


def vTransform(rot, p_rel=Translation()):
    """ Transform matrix of velcoity and angular (6x6) """
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


class ArmDynamic:
    def __init__(self, linkparam):
        self.linkparam = linkparam
        self.Ts0 = [Transform()]
        self.Ts = [Transform()]
        self.v = [vTranslate()]
        self.a = [vTranslate()]
        self.xy = [Translation()]
        self.ac = [Translation()]
        self.F = [Translation()]
        self.N = [Translation()]
        self.f = [Translation()] * (len(linkparam) + 1)
        self.n = [Translation()] * (len(linkparam) + 1)

    def addGravity(self, axis=2, g=9.8):
        self.a[0].mat[axis] = g

    def run(self, th, vth, ath):
        # joint
        for i in range(len(self.linkparam)):
            # set angle of joint
            self.linkparam[i][2] = th[i]

            # transform matrix for angle
            T = link(*self.linkparam[i])
            self.Ts.append(T)
            self.Ts0.append(self.Ts0[-1] * T)

            # Space location i+1
            self.xy.append(self.Ts0[-1] * Translation())

            # transform matrix for velcoity
            T_v = vTransform(Rotation(mat=T.rot.mat.T), T.loc)

            # velcoity i+1 (R-joint)
            new_v = T_v * self.v[i]
            new_v.mat[5] += vth[i] / 180 * np.pi
            self.v.append(new_v)

            # acceleration i+1 (R-joint)
            new_a = T_v * self.a[i]
            w = self.v[i].mat[3:6]
            new_a.mat[5] += ath[i] / 180 * np.pi
            new_a.mat[0:3] += T.rot.mat.T.dot(np.cross(w, np.cross(w, T.loc.mat)))
            new_a.mat[3:6] += T.rot.mat.T.dot(np.cross(w, [0, 0, vth[i] / 180 * np.pi]))
            self.a.append(new_a)

        # link
        for i in range(1, len(self.linkparam)):
            # acceleration i+1 on center(Center located at i+1)
            center = self.C[i]
            new_w = self.v[i].mat[3:6]
            new_a = self.a[i]
            new_ac = new_a.mat[0:3] + np.cross(new_w, np.cross(new_w, center)) + np.cross(new_a.mat[3:6], center)
            self.ac.append(new_ac)

            # force i+1 on center
            self.F.append(Translation(mat=new_ac * self.M[i]))

            # Torque i+1 on center
            now_I = np.array(self.I[i])
            self.N.append(Translation(mat=now_I.dot(new_a.mat[3:6]) + \
                                                    np.cross(new_w, now_I.dot(new_w))))


        for i in reversed(range(len(self.linkparam))):
            T = self.Ts[i + 1]
            now_F = self.F[i]
            self.f[i] = now_F + T.rot * self.f[i + 1]

            # torque on motor
            center = self.C[i]
            self.n[i] = self.N[i] + T.rot * self.n[i + 1] + Translation(mat=np.cross(center, now_F.mat) + \
                                                                            np.cross(T.loc.mat, (T.rot * self.f[i + 1]).mat))

        # remove last one
        self.f = self.f[:-1]
        self.n = self.n[:-1]

        return self.f, self.n


if __name__ == "__main__":
    pass
