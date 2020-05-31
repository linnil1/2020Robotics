import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from hw6 import Kalmen
from itertools import chain
from tqdm import tqdm


class GetHidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn1 = nn.Linear(4, 8)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.nn2 = nn.Linear(16, 32)
        # self.relu2 = nn.ReLU(inplace=True)
        self.nn2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.nn1(x)
        # x = self.relu1(x)
        x = self.nn2(x)
        # x = self.relu2(x)
        # x = self.nn3(x)
        return x

# train init
n = 100
threshold = 0.01
GH = GetHidden()
optimizer = torch.optim.Adam(chain(GH.parameters()), lr=0.001)
# optimizer = torch.optim.SGD(chain(GH.parameters()), lr=.1, momentum=0.01)
train_x = [[0, 0.8]] * n
train_loss = []

# real world
t = 0.001 
na = 0.1
nb = 0.0001
a = np.random.normal(scale=np.sqrt(na), size=n)
b = np.random.normal(scale=np.sqrt(nb), size=n)
real_X = [[0, 1]] * n

# observe 
observe_x = [0] * n
observe_v = [0] * n

# Kalmen model
kalmen = Kalmen()
G = np.array([t ** 2 / 2, t])
kalmen.H = Measure = np.array([[1, 0], [0, 0]])
kalmen.F = Forward = np.array([[1, t], [0, 1]])
kalmen.Na = Na = np.outer(G, G) * na
kalmen.Nb = Nb = np.array([[nb, 0], [0, 1e-18]])


def dot(t, a):
    return torch.matmul(torch.Tensor([t]), a.view(1, 2, 1))


# step
for i in tqdm(range(1, n)):
    # real
    real_X[i] = Forward.dot(real_X[i - 1]) + G * a[i]

    # observe without kalmen
    ox = real_X[i - 1][0] + b[i]
    observe_x[i] = ox
    observe_v[i] = (ox - observe_x[i - 1]) / t

    # kalmen
    kalmen.step(np.array([ox, 0]))

    # start from k state
    k = max(i - 5, 1)
    for _ in range(5):
        # Training
        loss = 0
        loss1 = 0
        loss2 = 0
        optimizer.zero_grad()
        old_x = torch.Tensor(train_x[k - 1])
        for j in range(k, i + 1):
            # state j
            pred_x = dot(Forward, old_x)
            obx = torch.Tensor([observe_x[j], 0]).view(1, 2, 1)
            new_x = GH(torch.cat([
                pred_x.flatten(),
                obx.flatten()
            ]).view(1, 4))
            # print("123")
            # print(pred_x.flatten())
            # print(new_x.flatten())

            # save state j
            train_x[j] = new_x[0].detach().numpy()
            old_x = new_x

            # v = (x1 - x0) /t
            loss1 += F.mse_loss(obx, dot(Measure, new_x))
            loss2 += F.mse_loss(pred_x[:, :, 0], new_x[:, :]) * 1.
            # print("00")
            # print((obx - dot(Measure, new_x)).flatten().detach().numpy())
            # print((pred_x[:, :, 0] - new_x[:, :]).detach().numpy())
            # loss2 = F.mse_loss(pred_x[:, 0, 0], new_x[:, 0]) * .001

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

    train_loss.append([loss1.detach().numpy(), loss2.detach().numpy()])

# to array
train_x = np.array(train_x)
train_loss = np.array(train_loss)
kalmen.summary()
real_X = np.array(real_X)

# plot
plt.subplot(2, 2, 1)
plt.title("Position")
plt.plot(observe_x, label="Observed with noise")
plt.plot(real_X[:, 0], label="real")
# plt.plot(kalmen.history_X[:, 0], label="kalmen", color='r')
plt.plot(train_x[:, 0], label="Train", color='r')
plt.xlabel("Time(ms)")
plt.ylabel("Position(m)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(observe_v, label="Observed with noise", color='lavender')
plt.plot(real_X[:, 1], label="real")
# plt.plot(kalmen.history_X[:, 1], label="kalmen", color='r')
plt.plot(train_x[:, 1], label="Train", color='r')
# plt.plot((train_x[1:, 0] - train_x[:-1, 0]) / t, label="Train", color='r')
plt.ylim([real_X[0][1] - 2, real_X[0][1] + 2])
plt.xlabel("Time(ms)")
plt.ylabel("velcoity(m/s)")
plt.title("Velcoity")
plt.legend()

plt.subplot(2, 2, 3)
plt.title("Error")
train_loss = np.array(train_loss)
# plt.plot(kalmen.history_P[:, 1, 1])
plt.plot(train_loss[:, 0], label="loss1")
plt.plot(train_loss[:, 1], label="loss2")
plt.legend()
plt.xlabel("Time(ms)")
plt.ylabel("Error")
plt.show()

exit()

plt.subplot(2, 2, 4)
plt.title("K")
plt.plot(kalmen.history_K[:, 0, 0])
plt.xlabel("Time(ms)")
plt.ylabel("Gain")
plt.show()
