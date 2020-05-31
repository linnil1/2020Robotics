import numpy as np
import matplotlib.pyplot as plt


class Kalmen:
    def __init__(self):
        """
        Kalmen Filter
        new_X = F * X + G * N_a
        Z = H * X + N_b

        X is state
        K is kalmen gain
        P is error
        N_a, N_b is noise and it's covariance is Na and Nb
        """
        self.clean()

    def clean(self):
        self.history_X = []
        self.history_K = []
        self.history_P = []
        self.X = np.zeros(2)
        self.K = np.identity(2)
        self.P = np.identity(2)

    def step(self, observe):
        # Corrector
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P).dot(self.H.T) + self.Nb))
        self.X = self.X + self.K.dot(observe - self.H.dot(self.X))
        self.P = (np.identity(2) - self.K.dot(self.H)).dot(self.P)

        # Predictor
        self.X = self.F.dot(self.X)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Na

        # save it
        self.history_X.append(self.X)
        self.history_K.append(self.K)
        self.history_P.append(self.P)

    def summary(self):
        self.history_X = np.array(self.history_X)
        self.history_K = np.array(self.history_K)
        self.history_P = np.array(self.history_P)


if __name__ == "__main__":
    # real world
    t = 0.001 
    n = 1000
    na = 0.1
    nb = 0.0001
    a = np.random.normal(scale=np.sqrt(na), size=n)
    b = np.random.normal(scale=np.sqrt(nb), size=n)
    real_X = [[0, 1]]

    # observe 
    observe_x = [0]
    observe_v = [0]

    # model
    H = np.array([[1, 0], [0, 0]])
    F = np.array([[1, t], [0, 1]])
    G = np.array([t ** 2 / 2, t])
    Na = np.outer(G, G) * na
    Nb = np.array([[nb, 0], [0, 1e-18]])

    # kalmen
    kalmen = Kalmen()
    kalmen.H = H
    kalmen.F = F
    kalmen.Na = Na
    kalmen.Nb = Nb

    for i in range(n):
        # real
        real_X.append(F.dot(real_X[-1]) + G * a[i])

        # observe without kalmen
        ox = real_X[-1][0] + b[i]
        observe_x.append(ox)
        observe_v.append((ox - observe_x[-2]) / t)

        # kalmen
        kalmen.step(np.array([ox, 0]))

    kalmen.summary()
    real_X = np.array(real_X)
    # plot
    plt.subplot(2, 2, 1)
    plt.title("Position")
    plt.plot(observe_x, label="without_kalmen")
    plt.plot(real_X[:, 0], label="real")
    plt.plot(kalmen.history_X[:, 0], label="kalmen", color='r')
    plt.xlabel("Time(ms)")
    plt.ylabel("Position(m)")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(observe_v, label="without_kalmen", color='lavender')
    plt.plot(real_X[:, 1], label="real")
    plt.plot(kalmen.history_X[:, 1], label="kalmen", color='r')
    plt.ylim([real_X[0][1] - 1, real_X[0][1] + 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("velcoity(m/s)")
    plt.title("Velcoity")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Error")
    plt.plot(kalmen.history_P[:, 1, 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("Error")

    plt.subplot(2, 2, 4)
    plt.title("K")
    plt.plot(kalmen.history_K[:, 0, 0])
    plt.xlabel("Time(ms)")
    plt.ylabel("Gain")
    plt.show()
