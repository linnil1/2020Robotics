import torch
import torch.nn.functional as F
import numpy as np
from hw2 import Transform, Translation, Rotation, Base
from hw3 import link, Quaternion


def tensorLink(twist, dist, angle, offset):
    """ Link made by Tensor """
    twist = torch.Tensor([twist / 180 * np.pi])
    T1 = torch.Tensor([
        [1., 0, 0, dist],
        [0, torch.cos(twist), -torch.sin(twist), 0],
        [0, torch.sin(twist),  torch.cos(twist), 0],
        [0, 0, 0, 1]])
    T2 = torch.tensor([
        [0., 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, offset],
        [0, 0, 0, 1]])
    # override by tensor required gradient
    T2[0][0] = T2[1][1] = torch.cos(angle)
    T2[1][0] = torch.sin(angle)
    T2[0][1] = -torch.sin(angle)
    return torch.mm(T1, T2)


def getInverseAngle(wanted, linkparam, init_angle=None):
    # init tensor
    if init_angle is None:
        th = torch.rand((6), requires_grad=True)
    else:
        th = torch.tensor(init_angle, requires_grad=True)
    wanted = torch.Tensor(wanted)
    optimizer = torch.optim.Adam([th], lr=0.2)

    # gradient decent
    for epoch in range(1000):
        optimizer.zero_grad()
        # th = torch.clamp(th, min=0, max=2 * np.pi)

        # forward the kinematic matrix
        T = torch.Tensor(np.identity(4))
        for i, p in enumerate(linkparam):
            T = torch.mm(T, tensorLink(p[0], p[1], th[i], p[3]))

        # calculate loss and back propagation
        loss = torch.mean((T[:3, :3] - wanted[:3, :3]) ** 2) * 1 + \
               torch.mean((T[:, 3] - wanted[:, 3]) ** 2) * 0.001
        loss.backward()
        optimizer.step()

        # show the training status
        if loss < 1e-6:
            break
        if epoch % 10 == 0:
            print(epoch, loss.item(), th.detach().numpy())

    return th.detach().numpy()


linkparam = [( 0,   0, None, 352),
             (90,  70, None, 0),
             ( 0, 360, None, 0),
             (90,   0, None, 380),
             (90,   0, None, 0),
             (90,   0, None, 65)]


def inverse_irb140(wanted, init_angle=None):
    th = getInverseAngle(wanted, linkparam, init_angle) / np.pi * 180
    T = forward_irb140(th)
    print("Angle", th)
    print("calc", T)
    print("wanted", wanted)
    return th, T


def forward_irb140(th):
    T = Transform()
    for i, p in enumerate(linkparam):
        T = T * link(p[0], p[1], th[i], p[3])
    return T


if __name__ == "__main__":
    """
    # Q7-1
    # Calculate Wanted Transform matrix
    T_wt = Transform(loc=Translation(5, 10, 200))
    T_bs = Transform(rot=Rotation(30, 2), loc=Translation(700, 0, 500))
    T_sg = Transform(rot=Rotation(60, 2), loc=Translation(30, 30, 30))
    T_bw = T_bs * T_sg * T_wt.T()
    inverse_irb140(T_bw)
    """

    # Q7-2
    T_bw1 = Transform(rot=Rotation(90, 0), loc=Translation(550, 0, 400))
    T_bw2 = Transform(rot=Rotation(90, 2) * Rotation(30, 1) * Rotation(45, 0), loc=Translation(650, 100, 300))
    print("T_bw1", T_bw1)
    print("T_bw2", T_bw2)
    T_w1w2 = T_bw1.T() * T_bw2
    print("T_w1w2", T_w1w2)

    r = T_w1w2.rot.mat
    angle = np.arccos((np.trace(r) - 1) / 2)
    raxis = 1 / 2 / np.sin(angle) * np.array([
        r[2,1] - r[1,2],
        r[0,2] - r[2,0],
        r[1,0] - r[0,1]])
    print("Rotation axis", raxis)
    print("Rotation angle", angle)

    w = Quaternion.setRotation(angle / np.pi * 180, raxis)
    m = w.getRotationMat()
    mat = w.T().getRotationMat()
    mat[1:4, 1:4] = mat[1:4, 1:4].T
    print(mat.dot(m))

    th1, T1 = inverse_irb140(T_bw1.mat)
    th2, T2 = inverse_irb140(T_bw2.mat, th1 / 180 * np.pi)

    print("Angle w1", th1)
    print("Angle w2", th2)
    print("Angle velocity", (th2 - th1) / 10)

    """
    # try rotate each joint in same rotation speed
    # but the result is not constant
    ths = np.linspace(th1, th2)
    Ts = np.stack([forward_irb140(th).mat for th in ths])
    pos_end = Ts[:, :3, 3]
    print(np.sum((pos_end[1:] - pos_end[:-1]) ** 2, axis=1))
    """
