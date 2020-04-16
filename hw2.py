import numpy as np

np.set_printoptions(3, suppress=True)


class Base:
    def __init__(self, n=3, mat=None):
        if mat is not None:
            self.mat = np.array(mat)
            return
        self.mat = np.identity(n)

    def __str__(self):
        return f"{self.mat}"

    def norm(self):
        return np.sqrt(np.sum(self.mat ** 2))

    def __add__(self, other):
        return self.__class__(mat=self.mat + other.mat)


class Rotation(Base):
    """ Rotation matrix """
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
    """ Rotation matrix """
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


if __name__ == "__main__":
    # Q3
    print("Q3")
    print(Rotation(30, 2) * Rotation(45, 0))

    # Q5
    print("Q5")
    print(Transform(rot=Rotation(30, 2), loc=Translation(0, 0, 0)) * Translation(10, 20, 30))
    print(Transform(rot=Rotation(30, 2), loc=Translation(11, -3, 9)) * Translation(10, 20, 30))
    print(Transform(rot=Rotation(30, 2), loc=Translation(11, -3, 9)).T())

    # Q6
    print("Q6")
    Tab = Transform(rot=Rotation(180, 2), loc=Translation(3, 0, 0))
    Tbc = Transform(rot=Rotation(90, 1) * Rotation(150, 0), loc=Translation(0, 0, 2))
    print("Tab", Tab)
    print("Tbc", Tbc)
    print("Tac", (Tab * Tbc))
    print("Tba", Tab.T())
    print("Tcb", Tbc.T())
    print("Tca", (Tab * Tbc).T())

    # Q8
    print("Q8")
    rot = Rotation(mat=np.matrix("0.866 -0.5 0; 0.433 0.75 -0.5; 0.25 0.433 0.866"))
    a, b, r = rot.getXYZ()
    print(rot)
    print("XYZ or ZYX abr:", a, b, r)
    print(Rotation(a, 2) * Rotation(b, 1) * Rotation(r, 0))
    a, b, r = rot.getZYZ()
    print("ZYZ abr:", a, b, r)
    print(Rotation(a, 2) * Rotation(b, 1) * Rotation(r, 2))

    # Q9
    th = 40 / 2 / 180 * np.pi
    e = [0, 1 / np.sqrt(2) * np.sin(th), 1 / np.sqrt(2) * np.sin(th), 0, np.cos(th)]
    R = np.array([
        [1 - 2*e[2]**2 - 2*e[3]**2, 2*(e[1]*e[2] - e[3]*e[4]), 2*(e[1]*e[3] + e[2]*e[4])],
        [2*(e[1]*e[2] + e[3]*e[4]), 1 - 2*e[1]**2 - 2*e[3]**2, 2*(e[2]*e[3] - e[1]*e[4])],
        [2*(e[1]*e[3] - e[2]*e[4]), 2*(e[1]*e[4] + e[2]*e[3]), 1 - 2*e[1]**2 - 2*e[2]**2]])
    print("Q9")
    print(R)
    p = Rotation(30, 2) * Rotation(20, 1) * Rotation(10, 0) * Rotation(mat=R)
    print(p)
    print(p.getXYZ())

    # Q10
    print("Q10")
    a = Transform(mat=np.matrix("0 1 0 1; 1 0 0 10; 0 0 -1 9; 0 0 0 1"))
    b = Transform(mat=np.matrix("1 0 0 -10; 0 -1 0 20; 0 0 -1 10; 0 0 0 1"))
    print(b.T() * a)
