import numpy as np
import torch
import torch.nn.functional as F
from simulation import Simulation
from gradient import tensorLink, forwardVariable
from hw2 import Transform, Translation, Rotation
import matplotlib.animation as animation
import matplotlib.pyplot as plt

target = Transform(rot=Rotation(90, 0), loc=Translation(550, 0, 400)).mat
linkparam = [( 0,   0, None, 352),
             (90,  70, None, 0),
             ( 0, 360, None, 0),
             (90,   0, None, 380),
             (90,   0, None, 0),
             (90,   0, None, 65)]
sim = Simulation(linkparam, 500)


def gradientVariable(wanted, linkparam, init_angle=None):
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
    pre_loss = torch.tensor(1e99)
    for epoch in range(500):
        sim.sim(th.detach().numpy() / np.pi * 180,
                text="loss : " + str(np.round(pre_loss.detach().numpy(), 5)))
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
        if loss <= pre_loss and pre_loss - loss < 1e-7:
            break
        pre_loss = loss
        if epoch % 10 == 0:
            print(epoch, loss.item(), th.detach().numpy())

    return np.mod(th.detach().numpy(), 2 * np.pi) / np.pi * 180


th = gradientVariable(target, linkparam)
print("Angle", th)
print("wanted", target)
print("calc", forwardVariable(th, linkparam))

ani = sim.runAnimation(show=True)
# ani.save('visualize_gradient.gif', dpi=80, writer='imagemagick')
