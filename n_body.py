import numpy as np
from vispy import app
from vispy.scene import SceneCanvas
from vispy.scene.visuals import XYZAxis, Markers
from vispy.visuals.transforms import MatrixTransform
from vispy.scene.cameras import FlyCamera


class NewtonMatrix(SceneCanvas):
    """

    """
    def __init__(self, num_bods=100, mass=None, pos_0=None, vel_0=None, *args, **kwargs):
        """

        :type T0: np.float64
        :param num_bods:
        """
        global norm_pos, mag_pos
        super(NewtonMatrix, self).__init__(title="Figuring out Markers and SHit", keys='interactive',
                                           fullscreen=True, size=(800, 600))
        self.unfreeze()
        self.warp = .5
        self.view = self.central_widget.add_view()
        self.view.camera = "turntable"
        self.zero = np.zeros((1, 3), dtype=np.float64)
        G = 6.67430e-11
        self.G = G * 500
        self.T0 = 0
        self.T_MAX = 1000
        self.N_BODS = num_bods
        self.M_hist = []
        self.avg_pos = self.zero.copy()
        self.avg_vel = self.zero.copy()
        self.avg_acc = self.zero.copy()

        if mass is None:
            self.mass = np.ndarray((self.N_BODS, 1), dtype=np.float64)
            # put a randomized distribution of masses here
            self.mass = np.random.normal(loc=2000, scale=400, size=self.N_BODS)
            self.mass[0] *= 1000
            # print("N_BODS =", self.N_BODS)
            # print("MASS =", self.mass)
        else:
            self.mass = mass

        if pos_0 is None:
            self.pos_0 = np.zeros((1, self.N_BODS), dtype=type(np.array(3, dtype=np.float64)))
            norm_pos = np.zeros((1, self.N_BODS), dtype=type(np.array(3, dtype=np.float64)))
            # put a randomized distribution of positions here
            _th = np.linspace(0, 2 * np.pi, self.N_BODS)
            mag_pos = np.random.normal(loc=500, scale=100, size=self.N_BODS)
            # print("mag_pos =", mag_pos, len(mag_pos))
            norm_pos = np.array([np.cos(_th), np.sin(_th), np.sin(_th - _th)]).transpose()
            # print("norm_pos =", norm_pos, len(norm_pos))
            self.pos_0 = np.array([mag_pos[n] * norm_pos[n] for n in range(0, self.N_BODS)])
            # print("pos_0 =", self.pos_0, len(self.pos_0))
            self.pos_0[0] = self.zero.copy()
        else:
            self.pos_0 = pos_0

        if vel_0 is None:
            self.vel_0 = np.zeros((1, self.N_BODS), dtype=type(np.array(3, dtype=np.float64)))
            # put a randomized distribution of velocities here
            dv = np.array(np.sin(np.random.normal(loc=0, scale=15 * np.pi / 180, size=self.N_BODS)))
            # print("dv =", dv)
            self.vel_0 = [norm_pos[n] * dv[n] + np.cross([0, 0, 1], norm_pos[n]) / np.sqrt(mag_pos[n]) for n in range(0, self.N_BODS)]
            # self.vel_0[0] = self.zero.copy()
        else:
            self.vel_0 = vel_0

        assert (len(self.mass) == self.N_BODS)
        assert (len(self.pos_0) == self.N_BODS)
        assert (len(self.vel_0) == self.N_BODS)

        # [print(self.mass[n], self.pos_0[n], self.vel_0[n]) for n in range(0, self.N_BODS - 1)]
        self.SM = np.ndarray((self.N_BODS + 1, self.N_BODS + 1, 3), dtype=np.float64)
        # print(type(self.SM), "\n")
        # self.SM[0, 0] = self.T0
        # for n in range(0, self.N_BODS - 1):
        #     self.SM[1 + n, 0] = self.pos_0[n]
        #     self.SM[0, 1 + n] = self.vel_0[n]
        # print(type(self.SM[1, 0]), type(self.SM[0, 1]))

        self.particles = Markers(parent=self.view.scene)
        self.view.add(self.particles)

        self.SM[0, 0] = self.T0
        dat = []
        for n in range(0, self.N_BODS - 1):
            self.SM[1 + n, 0] = self.pos_0[n]
            self.SM[0, 1 + n] = self.vel_0[n]
            dat.append(self.pos_0[n])
        # print(dat)
        self.set_rel_posvel()
        self.set_accel()
        self.get_averages()
        self.particles.set_data(pos=np.array(dat), size=4, edge_color="red", edge_width=1)
        self.show()
        self._timer = app.Timer(interval='auto', connect=self.iterate, start=True, app=self.app)
        self.freeze()

# =========================================================================================================

    def set_rel_posvel(self):
        """
        :return:
        """
        for i in range(1, self.N_BODS + 1):
            for j in range(1, self.N_BODS + 1):
                if i == j:
                    # print("(i=j): ZERO???", self.SM[j, 0], self.SM[i, 0])
                    self.SM[i, j] = self.zero
                elif i > j:
                    # print("(i<j): ", self.SM[j, 0], self.SM[i, 0])
                    self.SM[i, j] = self.SM[j, 0] - self.SM[i, 0]
                elif i < j:
                    # print("(i>j): ", self.SM[0, j], self.SM[0, i])
                    self.SM[i, j] = self.SM[0, j] - self.SM[0, i]

    def set_accel(self):
        """
        :return:
        """
        for j in range(1, self.N_BODS):
            accel = self.zero
            for i in range(1, self.N_BODS + 1):

                if i > j:
                    dist_sqr = np.linalg.norm(self.SM[i, j], axis=0)
                    dist_sqr *= dist_sqr
                    # print(i, j, ":dist_sqr (i>j) = ", dist_sqr)
                    if dist_sqr != 0:
                        accel += self.warp * self.SM[i, j] * (-self.G * self.mass[j] / dist_sqr)
                elif i < j:
                    dist_sqr = np.linalg.norm(self.SM[j, i], axis=0)
                    dist_sqr *= dist_sqr
                    # print(i, j, ":dist_sqr (i<j) = ", dist_sqr)
                    if dist_sqr != 0:
                        accel += -self.warp * self.SM[j, i] * (-self.G * self.mass[j] / dist_sqr)
                elif i == j:
                    dist_sqr = 0.0
                    # print(i, j, ":dist_sqr (i=j) = ", dist_sqr)

            if j != 1:
                self.SM[j, j] = accel - self.avg_acc
            else:
                self.SM[j, j] = self.zero.copy()
        # [print("accel[", n, "] =", self.SM[n, n]) for n in range(1, self.N_BODS)]

    def iterate(self, event):
        """ Use acceleration values to update vel, then pos."""
        # self.M_hist.append(self.SM)
        if self.T0 < self.T_MAX:
            self.T0 += 1
        else:
            pass
        dat = [self.SM[1, 0]]
        for n in range(1, self.N_BODS):
            self.SM[0, 0] = self.T0
            self.SM[0, n + 1] += self.SM[n + 1, n + 1]
            self.SM[n + 1, 0] += self.SM[0, n + 1]
            # self.SM[n + 1, 0] -= self.avg_pos
            dat.append(self.SM[n + 1, 0])

        self.particles.set_data(pos=np.array(dat), size=4, edge_color="red", edge_width=1)
        self.set_rel_posvel()
        self.set_accel()

    def get_averages(self):
        self.avg_pos = self.zero.copy()
        self.avg_vel = self.zero.copy()
        self.avg_acc = self.zero.copy()
        for n in range(1, self.N_BODS + 1):
            self.avg_pos += self.SM[n, 0]
            self.avg_vel += self.SM[0, n]
            self.avg_acc += self.SM[n, n]
        self.avg_pos /= self.N_BODS
        self.avg_vel /= self.N_BODS
        self.avg_acc /= self.N_BODS
        print("\n>>>>", self.T0, "<<<<")
        print("AVG_POS =", self.avg_pos)
        print("AVG_VEL =", self.avg_vel)
        print("AVG_ACC =", self.avg_acc)

    # def on_timer(self, event):
    #     data = self.M_hist[self.T0]
    #     self.set_data(pos=data[1:, 0], size=3, edge_color="red")
    #     self.update()


def axis_visual(scale=1.0, parent=None):
    """
    Returns a :class:`vispy.scene.visuals.XYZAxis` class instance using given
    scale.

    Parameters
    ----------
    scale : numeric, optional
        Axis visual scale.
    parent : Node, optional
        Parent of the axis visual in the `SceneGraph`.

    Returns
    -------
    XYZAxis
        Axis visual.
    """

    axis = XYZAxis(parent=parent)

    transform = MatrixTransform()
    transform.scale((scale, scale, scale))
    axis.transform = transform

    return axis


def main():
    can = NewtonMatrix()
    axis = axis_visual(scale=1000)
    can.view.add(axis)
    can.show()
    can.app.run()

if __name__ == '__main__':
    main()
