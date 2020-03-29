import numpy as np
from hw2 import Transform, Translation, Rotation, Base
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def testQuaternionRotation():
    """ Test quaternino operation is same to analytical method """
    rot = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    theta = 40
    pos = [-0.1, 0.5, np.sqrt(1 - 0.1 ** 2 - 0.5 ** 2)]
    w = Quaternion.setRotation(theta, rot)
    m = w.getRotationMat()
    mat = w.T().getRotationMat()
    mat[1:4, 1:4] = mat[1:4, 1:4].T
    print(mat.dot(m))
    print((w * Quaternion(0, *pos) * w.T()).getIJK())

    th = 40 / 2 / 180 * np.pi
    e = [0, *(rot * np.sin(th)), np.cos(th)]
    R = np.array([
        [1 - 2*e[2]**2 - 2*e[3]**2, 2*(e[1]*e[2] - e[3]*e[4]), 2*(e[1]*e[3] + e[2]*e[4])],
        [2*(e[1]*e[2] + e[3]*e[4]), 1 - 2*e[1]**2 - 2*e[3]**2, 2*(e[2]*e[3] - e[1]*e[4])],
        [2*(e[1]*e[3] - e[2]*e[4]), 2*(e[1]*e[4] + e[2]*e[3]), 1 - 2*e[1]**2 - 2*e[2]**2]])
    print(R)
    print(Rotation(mat=R) * Translation(mat=pos))


def findTransform(r1, r2):
    """ Find the translation, scaling, rotation of r1 from r2 """
    # normalize
    rr1 = r1 - r1.mean(axis=1)[:, None]
    rr2 = r2 - r2.mean(axis=1)[:, None]

    # find scale
    scale = np.sqrt((rr1 ** 2).sum(axis=0) / (rr2 ** 2).sum(axis=0)).mean()
    rr2 *= scale

    # find rotation
    ## calc convariance matrix
    cov = rr2.dot(rr1.T)
    M = np.array([
        [cov[0,0]+cov[1,1]+cov[2,2], 0, 0, 0],
        [cov[1,2]-cov[2,1], cov[0,0]-cov[1,1]-cov[2,2], 0, 0],
        [cov[2,0]-cov[0,2], cov[0,1]+cov[1,0], cov[1,1]-cov[0,0]-cov[2,2], 0],
        [cov[0,1]-cov[1,0], cov[2,0]+cov[0,2], cov[2,1]+cov[1,2], cov[2,2] - cov[0,0] - cov[1,1]]
        ])
    for i in range(0, 3):
        for j in range(i + 1, 4):
            M[i, j] = M[j, i]

    ## calc rotation matrix
    eigen = np.linalg.eig(M)
    q = Quaternion(mat=eigen[1][:, 0])
    # a = q.getRotationParam()
    # q = Quaternion.setRotation(a[0], a[1])
    qmat = q.getRotationMat()
    qmat_inv = q.T().getRotationMat()
    qmat_inv[1:4, 1:4] = qmat_inv[1:4, 1:4].T
    R = qmat.dot(qmat_inv)[1:, 1:]

    # find translation
    translate = np.mean(r1 - R.dot(r2) * scale, axis=1)

    # print the parameters
    print("---" * 20)
    print("Eigen", eigen)
    print("translation", translate)
    print("scale", scale)
    print("Rotation", R)
    print("Rotation param", q.getRotationParam())

    s = Transform()
    s.mat[np.arange(3), np.arange(3)] = scale
    transform = Transform(loc=Translation(mat=translate), rot=Rotation(mat=R)) * s
    print("Transform matrix", transform)
    return transform


if __name__ == "__main__":
    data = scipy.io.loadmat('hw3-data.mat')
    r1 = data['r1']
    r2 = data['r2']
    transform = findTransform(r1, r2)

    # diff r1 r2
    r2_new = transform.mat.dot(np.vstack([r2, np.ones([1, r2.shape[1]])]))[:3]
    print("r1", r1)
    print("r2", r2_new)
    print(((r1 - r2_new) ** 2).sum(axis=0).mean())

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r1[0, :],     r1[1, :],     r1[2, :],     marker='o')
    ax.scatter(r2_new[0, :], r2_new[1, :], r2_new[2, :], marker='^')
    plt.show()

    # testQuaternionRotation()
