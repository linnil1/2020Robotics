import numpy as np
import matplotlib.pyplot as plt

# real world
x = 0
v = 1
t = 0.001 
n = 1000
na = 0.1
nb = 0.0001
a = np.random.normal(scale=np.sqrt(na), size=n)
b = np.random.normal(scale=np.sqrt(nb), size=n)
real_x = [x]
real_v = [v]

# observe 
ox = 0
ov = 0
observe_x = [ox]
observe_v = [ov]

# kalmen
H = np.array([[1, 0], [0, 0]])
F = np.array([[1, t], [0 ,1]])
X = np.zeros(2)
kalmen_x = [X[0]]
kalmen_v = [X[1]]
K = np.identity(2)
P = np.identity(2)
Na = np.outer(np.array([t ** 2 / 2, t]), np.array([t ** 2 / 2, t])) * na
Nb = np.array([[nb, 0], [0, 1e-18]])
kalmen_k = [K]
kalmen_p = [P]

for i in range(n):
    # real world
    dx = v * t
    v += a[i] * t
    x += dx
    real_x.append(x)
    real_v.append(v)

    # observe without kalmen
    observe_v.append((dx + b[i]) / t)
    ox = x + b[i]
    observe_x.append(ox)

    # kalmen
    K = P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + Nb))
    X = X + K.dot(np.array([ox, 0]) - H.dot(X))
    P = (np.identity(2) - K.dot(H)).dot(P)
    X = F.dot(X)
    P = F.dot(P).dot(F.T) + Na
    kalmen_x.append(X[0])
    kalmen_v.append(X[1])
    kalmen_k.append(K)
    kalmen_p.append(P)


# plot
kalmen_k = np.array(kalmen_k)
kalmen_p = np.array(kalmen_p)
plt.subplot(2, 2, 1)
plt.title("Position")
plt.plot(observe_x, label="without_kalmen")
plt.plot(real_x, label="real")
plt.plot(kalmen_x, label="kalmen", color='r')
plt.xlabel("Time(ms)")
plt.ylabel("Position(m)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(observe_v, label="without_kalmen", color='lavender')
plt.plot(real_v, label="real")
plt.plot(kalmen_v, label="kalmen", color='r')
plt.ylim([v - 1, v + 1])
plt.xlabel("Time(ms)")
plt.ylabel("velcoity(m/s)")
plt.title("Velcoity")
plt.legend()

plt.subplot(2, 2, 3)
plt.title("Error")
plt.plot(kalmen_p[:, 1, 1])
plt.xlabel("Time(ms)")
plt.ylabel("Error")

plt.subplot(2, 2, 4)
plt.title("K")
plt.plot(kalmen_k[:, 0, 0])
plt.xlabel("Time(ms)")
plt.ylabel("Gain")
plt.show()
