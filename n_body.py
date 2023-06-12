import logging
import numpy as np
from vispy import app
from vispy.scene import SceneCanvas
from vispy.scene.visuals import XYZAxis, Markers
from vispy.visuals.transforms import MatrixTransform
from vispy.scene.cameras import FlyCamera

logging.basicConfig(filename='n_body.log',
                    level=logging.INFO,
                    format='PV_%(levelname)s:%(asctime)s:%(message)s'
                    )


class NewtonMatrix(SceneCanvas):
    """

    """
    def __init__(self, num_bods=50, mass=None, pos_0=None, vel_0=None, *args, **kwargs):
        """

        :type T0: np.float64
        :param num_bods:
        """
        global norm_pos, mag_pos
        super(NewtonMatrix, self).__init__(title="Figuring out Markers and SHIT", keys='interactive',
                                           fullscreen=True, size=(800, 600))
        self.unfreeze()
        self.warp = 10
        self.view = self.central_widget.add_view()
        self.view.camera = "arcball"
        self.zero = np.zeros(3, dtype=np.float64)
        G = 6.67430e-11
        self.G = G * 1
        self.T0 = 0
        self.T_MAX = 1000
        self.N_BODS = num_bods
        self.M_hist = []
        self.avg_pos = self.zero.copy()
        self.avg_vel = self.zero.copy()
        self.avg_acc = self.zero.copy()
        self.cm_pos = self.zero.copy()

        if mass is None:
            # self.mass = np.ndarray((self.N_BODS, 1), dtype=np.float64)
            # put a randomized distribution of masses here
            self.mass = np.random.normal(loc=1000, scale=20, size=self.N_BODS)
            self.mass[0] *= 1e+09
            self.tot_mass = np.sum(self.mass)
            self.part_mass = self.mass / self.tot_mass
            # print("N_BODS =", self.N_BODS)
            # print("MASS =", self.mass)
        else:
            self.mass = mass

        if pos_0 is None:
            self.pos_0 = np.zeros(self.N_BODS, dtype=type(np.array(3, dtype=np.float64)))
            norm_pos = np.zeros(self.N_BODS, dtype=type(np.array(3, dtype=np.float64)))
            # put a randomized distribution of positions here
            thetas = np.linspace(0, 2 * np.pi, self.N_BODS)
            _ph = 0
            mag_pos = np.random.normal(loc=200, scale=50, size=self.N_BODS)
            # print("mag_pos =", mag_pos, len(mag_pos))
            norm_pos = np.array([(np.cos(t), np.sin(t), np.sin(_ph)) for t in thetas]) # , dtype=type(np.array(3, dtype=np.float64)))
            mag_pos[0] = 0
            # print("norm_pos =", norm_pos, len(norm_pos))
            self.pos_0 = [mag_pos[n] * norm_pos[n] for n in range(0, self.N_BODS)]
            # self.pos_0 = np.array([mag_pos[n] * norm_pos[n] for n in range(0, self.N_BODS)])
            # print("pos_0 =", self.pos_0, len(self.pos_0))
        else:
            self.pos_0 = pos_0

        if vel_0 is None:
            self.vel_0 = np.zeros(self.N_BODS, dtype=type(np.array(3, dtype=np.float64)))
            # put a randomized distribution of velocities here
            dv = np.array(np.sin(np.random.normal(loc=0, scale=15 * np.pi / 180, size=self.N_BODS)))
            # print("dv =", dv)
            self.vel_0 = [0]
            self.vel_0.extend([(1 + dv[n]) * norm_pos[n] + np.cross([0, 0, 1], norm_pos[n]) * np.sqrt(self.G * self.mass[0] / mag_pos[n]) for n in range(1, self.N_BODS)])
        elif vel_0 == 0:
            self.vel_0 = np.zeros(self.N_BODS, dtype=type(np.array(3, dtype=np.float64)))
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
        dat = np.zeros((self.N_BODS, 3), dtype=np.float64)
        for n in range(0, self.N_BODS):
            if n == 0:
                self.SM[1 + n, 0] = self.zero
                self.SM[0, 1 + n] = self.zero
            else:
                self.SM[1 + n, 0] = self.pos_0[n]
                self.SM[0, 1 + n] = self.vel_0[n]

        dat = self.SM[1:, 0]
        [logging.info(str(n) + ":: mass :" + str(self.mass[n]) + " *** " + str(type(dat[n])) +
                      " *** " + str(dat[n]) + "\n" + str(len(dat[n]))) for n in range(0, self.N_BODS)]
        print(dat.shape)
        self.set_rel_posvel()
        self.set_accel()
        self.get_averages()
        self.particles.set_data(pos=self.SM[1:, 0], size=2, edge_color="red", edge_width=1)
        self.show()
        self._timer = app.Timer(interval='auto', connect=self.iterate, start=True, app=self.app)
        self.freeze()

    # def set_posvel(self):
    #     self.SM[0, 0] = self.T0
    #     for n in range(0, self.N_BODS - 1):
    #         self.SM[1 + n, 0] = self.pos_0[n]
    #         self.SM[0, 1 + n] = self.vel_0[n]

    def set_rel_posvel(self):
        """
        :return:
        """
        for i in range(1, self.N_BODS + 1):
            for j in range(1, self.N_BODS + 1):
                if i == j:          # rel_pos and rel_vel both zero
                    # print("(i=j): ZERO???", self.SM[j, 0], self.SM[i, 0])
                    self.SM[i, j] = self.zero
                else:         # records rel position from i to j
                    # print("(i<j): ", self.SM[j, 0], self.SM[i, 0])
                    self.SM[i, j] = self.SM[j, 0] - self.SM[i, 0]
                # elif i < j:         # records rel velocity from i to j
                #     # print("(i>j): ", self.SM[0, j], self.SM[0, i])
                #     self.SM[i, j] = self.SM[0, i] - self.SM[0, j]

    def set_accel(self):
        """
        :return:
        """
        for j in range(1, self.N_BODS + 1):
            accel = self.zero
            for i in range(1, self.N_BODS + 1):
                if i != j:
                    dist_sqr = np.linalg.norm(self.SM[i, j])
                    dist_sqr *= dist_sqr
                    # print(i, j, ":dist_sqr (i>j) = ", dist_sqr)
                    if dist_sqr != 0:
                        accel += self.SM[i, j] * (-self.G * self.mass[j - 1] / dist_sqr)

            self.SM[j, j] = accel
        self.SM[1, 1] = np.array([0, 0, 0])
        # [print("accel[", n, "] =", self.SM[n, n]) for n in range(1, self.N_BODS)]

    def iterate(self, event):
        """ Use acceleration values to update vel, then pos."""
        # self.M_hist.append(self.SM)
        # if self.T0 < self.T_MAX:
        #     self.T0 += 1
        # else:
        #     pass
        self.T0 += 1
        self.SM[0, 0] = self.T0
        dat = []
        for n in range(1, self.N_BODS + 1):
            self.SM[0, n] += self.SM[n, n] * self.warp
            # self.SM[0, n] -= self.avg_acc
            # print(self.cm_pos.shape, self.SM[n, 0].shape)
            self.SM[n, 0] += self.SM[0, n] * self.warp
            # self.SM[n, 0] -= self.SM[1, 0]
            # self.SM[n, 0] -= self.cm_pos
            dat.append(self.SM[n, 0] - self.SM[1, 0])
        self.SM[1, 0] = self.zero.copy()
        print(type(dat))
        self.particles.set_data(pos=np.array(dat), size=4, edge_color="red", edge_width=1)
        trx = MatrixTransform()
        trx.rotate(-90, [1, 0, 0])
        trx.translate(-self.cm_pos)
        self.particles.transform = trx
        self.set_rel_posvel()
        self.set_accel()
        self.get_averages()

    def get_averages(self):
        self.avg_pos = self.zero.copy()
        self.avg_vel = self.zero.copy()
        self.avg_acc = self.zero.copy()
        self.cm_pos = self.zero.copy()
        tot_mass = 0
        for n in range(1, self.N_BODS + 1):
            self.avg_pos += self.SM[n, 0]
            self.avg_vel += self.SM[0, n]
            self.avg_acc += self.SM[n, n]
            self.cm_pos += self.part_mass[n - 1] * self.SM[n, 0]
        # self.cm_pos /= tot_mss
        # self.avg_pos /= self.N_BODS
        # self.avg_vel /= self.N_BODS
        # self.avg_acc /= self.N_BODS
        print("\n>>>>", self.T0, "<<<<")
        print("AVG_POS =", self.avg_pos)
        print("AVG_VEL =", self.avg_vel)
        print("AVG_ACC =", self.avg_acc)
        print("CENTER OF MASS =", self.cm_pos)


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

# def update():
#     global matrix, t, data
#     data = matrix.M_hist[t]
#     matrix.set_data(pos=np.array([data[n, 0] for n in range(1, matrix.N_BODS + 1)]), size=4, edge_color="red")
#     # matrix.update()
#     t += 1
#     t = t % 999


def main(viz=None):
    # print("MAIN")

    can = NewtonMatrix()
    # view.camera = "arcball"
    #
    # matrix = NewtonMatrix()
    # view.add(matrix)
    # timer = app.Timer(iterations=matrix.T_MAX, interval=1)
    # timer.connect(matrix.iterate())
    # timer.start(0)
    # view.camera.set_range()
    # timer = app.Timer(connect=matrix.on_timer, iterations=matrix.T_MAX, start = True)
    frame = axis_visual(scale=1000000, parent=can.view.scene)
    trx = MatrixTransform()
    trx.rotate(-90, [1, 0, 0])
    trx.scale((1, 1))
    frame.transform = trx
    can.view.add(frame)
    # can.particles.parent=frame
    can.show()
    can.app.run()


# def iter_matrix():
#     # N = 10
#     # MAX_T = 10
#     # T = 0
#     # mass = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., ]
#     # theta = np.linspace(0, 359, N)
#     # radius = np.random.normal(loc=1000, scale=100, size=N)
#     # pos_0 = np.zeros((N, 3), dtype=np.float64)
#     # vel_0 = np.zeros((N, 3), dtype=np.float64)
#     # vel0_dir = np.zeros((N, 3), dtype=np.float64)
#
#     # for n in range(0, N - 1):
#     #     r = radius[n]
#     #     ct = np.cos(theta[n])
#     #     st = np.sin(theta[n])
#     #     print(r, ct, st)
#     #     pos_0[n] = [r * ct, r * st, 0]
#     #     vel0_dir[n] = np.cross(pos_0[n], [0.0, 0.0, 1.0]) / np.linalg.norm(pos_0[n])
#     #     vel_0[n] = vel0_dir[n] * (1 / np.sqrt(np.linalg.norm(pos_0[n])))
#
#
#
#     while NM.T0 < NM.T_MAX:
#         NM.iterate()
#
#     return NM
#
#     print(np.array(NM.M_hist).shape)


if __name__ == '__main__':
    main()