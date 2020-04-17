import torch
import torch.nn.functional as F
import numpy as np
from hw2 import Transform, Translation, Rotation, Base
from hw3 import link, Quaternion


def tensorLink(twist, dist, angle, offset):
    """ Link made by Tensor """
    twist = torch.Tensor([twist / 180 * np.pi])
    T1 = torch.Tensor([
        [1, 0, 0, dist],
        [0, torch.cos(twist), -torch.sin(twist), 0],
        [0, torch.sin(twist),  torch.cos(twist), 0],
        [0, 0, 0, 1.]])
    T2 = torch.tensor([
        [0, 0, 0, 0.],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    # override by tensor required gradient
    T2[0][0] = T2[1][1] = torch.cos(angle)
    T2[1][0] = torch.sin(angle)
    T2[0][1] = -torch.sin(angle)
    T2[2][3] = offset
    return torch.mm(T1, T2)


def gradientVariable(wanted, linkparam, init_angle=None, iters=500):
    """
    Use gradient descent to find parameters
    Unknown parameters are setted to None.
    Assume no two parameters exist at the same link
    """
    # init tensor
    n = np.sum([p[2] is None or p[3] is None for p in linkparam])
    if init_angle is None:
        th = torch.rand((n), requires_grad=True)
    else:
        th = torch.tensor(init_angle, requires_grad=True)
    wanted = torch.Tensor(wanted)
    optimizer = torch.optim.Adam([th], lr=0.2)

    # gradient decent
    for epoch in range(iters):
        optimizer.zero_grad()

        # forward the kinematic matrix
        T = torch.Tensor(np.identity(4))
        i = 0
        for p in linkparam:
            if p[2] is None:
                T = torch.mm(T, tensorLink(p[0], p[1], th[i], p[3]))
                i += 1
            elif p[3] is None:
                T = torch.mm(T, tensorLink(p[0], p[1], p[2], th[i]))
                i += 1
            else:
                tmp_twist = torch.Tensor([p[2]]) / 180 * np.pi
                T = torch.mm(T, tensorLink(p[0], p[1], tmp_twist, p[3]))

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

    return np.mod(th.detach().numpy(), 2 * np.pi) / np.pi * 180


def forwardVariable(th, linkparam):
    T = Transform()
    i = 0
    for p in linkparam:
        if p[2] is None:
            T = T * link(p[0], p[1], th[i], p[3])
            i += 1
        elif p[3] is None:
            T = T * link(p[0], p[1], p[2], th[i])
            i += 1
        else:
            T = T * link(p[0], p[1], p[2], p[3])
    return T


if __name__ == "__main__":
    linkparam = [(90,   0, 90,  20),
                 ( 0,  20, None, 0),
                 ( 0, 100, None, 0),
                 ( 0, 100, None, 0),
                 ( 0,  60,   90, 0),
                 (90,   0,    0, 0)]
    wanted = [[  0.6,  0.,   0.8, 30. ],
              [  0. , -1.,  -0. ,  0. ],
              [  0.8,  0.,  -0.6, 140.],
              [  0. ,  0.,   0. ,  1. ]]
    th = gradientVariable(wanted, linkparam)
    print("Angle", th)
    print("wanted", np.array(wanted))
    print("calc", forwardVariable(th, linkparam))
