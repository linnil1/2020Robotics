import numpy as np
import matplotlib.pyplot as plt
from utils import Transform, Translation, link
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


class Simulation:
    def __init__(self, linkparam, limit=4):
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        self.limit = limit
        self.poses = []
        self.dirs = []
        self.trans = []
        self.text = []
        self.linkparam = linkparam

    def initPlot(self, title=''):
        plt.cla()
        limit = self.limit
        self.ax.set_xlim3d([-limit, limit])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-limit, limit])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-limit, limit])
        self.ax.set_zlabel('Z')
        plt.title(title)

    def sim(self, th, text=""):
        # init
        trans = [Transform()]
        pos = [Translation()]
        dirs = [np.identity(4)]

        # main
        for i in range(len(th)):
            L = self.linkparam[i]
            T = link(L[0], L[1], th[i], L[3])
            trans.append(trans[-1] * T)
            pos.append(trans[-1] * pos[0])
            dirs.append(trans[-1].mat)

        # save it
        self.trans.append(trans)
        self.poses.append(np.array([p.mat for p in pos]))
        self.dirs.append(dirs)
        self.text.append(text)

    def showPos(self, frame, poses, dirs):
        self.initPlot(str(frame))
        p = poses[frame]
        dirs = np.array(dirs[frame]) * self.limit / 5

        # old arm
        for i in range(0, frame):
            self.ax.plot(poses[i][:, 0], poses[i][:, 1], poses[i][:, 2],
                         "o-", color="gray")

        # new arm
        self.ax.plot(p[:, 0], p[:, 1], p[:, 2], "o-")
        for i in range(len(p)):
            self.ax.text(*p[i, :], str(i))

        # Direction
        self.ax.quiver(p[:, 0], p[:, 1], p[:, 2],
                       dirs[:, 0, 0], dirs[:, 1, 0], dirs[:, 2, 0],
                       color='blue')
        self.ax.quiver(p[:, 0], p[:, 1], p[:, 2],
                       dirs[:, 0, 1], dirs[:, 1, 1], dirs[:, 2, 1],
                       color='green')
        self.ax.quiver(p[:, 0], p[:, 1], p[:, 2],
                       dirs[:, 0, 2], dirs[:, 1, 2], dirs[:, 2, 2],
                       color='orange')

        # text
        self.ax.text2D(0.1, 0.9, self.text[frame],
                       transform=self.ax.transAxes)

    def runAnimation(self, repeat=True, show=True):
        line_ani = animation.FuncAnimation(self.fig,
                                           self.showPos,
                                           len(self.poses),
                                           fargs=(self.poses, self.dirs),
                                           interval=100,
                                           repeat=repeat)
        if show:
            plt.show()
        return line_ani


if __name__ == "__main__":
    linkparam = [( 0,   0, None, 352),
                 (90,  70, None, 0),
                 ( 0, 360, None, 0),
                 (90,   0, None, 380),
                 (90,   0, None, 0),
                 (90,   0, None, 65)]
    sim = Simulation(linkparam, 400)
    th = np.array([0, 0, 0, 0, 0, 0])

    for i in range(50):
        sim.sim(th)
        th[1] += 1
        th[0] -= 1
    sim.runAnimation()
