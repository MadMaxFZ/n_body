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

    def __init__(self, num_bods=50, mass=None, pos_0=None, vel_0=None,
                 t_start=0, t_end=100, n_frames=100, has_star=True, *args, **kwargs):
        """

        :type T0: np.float64
        :param num_bods:
        """
        global norm_pos, mag_pos
        super(NewtonMatrix, self).__init__(title="Figuring out Markers, Numpy, Transforms and SHIT",
                                           keys='interactive',
                                           fullscreen=True, size=(800, 600))
        self.unfreeze()
        G_base = 6.67430e-11                        # gravitational constant
        G_tweak = 1.0                               # tweak gravity by this factor
        self.G = G_base * G_tweak                   #
        self.N_frames = n_frames                    # number of snapshots over time interval
        self.ticks = 0
        self.warp = 0.01                            # 'time' elapsed per frame
        self.T_0 = t_start                          # 'time' at start of simulation
        self.T_1 = t_end                            # 'time' at end of simulation
        self.T_n = np.linspace(start=self.ticks,
                               stop=self.N_frames,
                               num=self.N_frames,
                               )
        if has_star:                                # if central mass desired,
            num_bods += 1                           #   add another spot for it

        self.N_bods = num_bods                      # number of particles to create
        self.zero = np.zeros(3, dtype=np.float64)   # default zero vector
        self.cm_pos = self.zero.copy()              # center of mass position
        self.cm_vel = self.zero.copy()              # center of mass velocity
        self.tot_mv = self.zero.copy()              # total linear momentum
        self.tot_w  = self.zero.copy()              # total angular momentum

        self.mass   = np.zeros(self.N_bods,
                               dtype=np.float64)

        self.accel  = np.zeros((self.N_frames,
                                self.N_bods,
                                3), dtype=np.float64)

        self.matrix = np.zeros((self.N_frames,
                                self.N_bods + 1,
                                self.N_bods + 1,
                                3), dtype=np.float64)

        self.frame_t = self.matrix[:, 0, 0]
        self.frame_t = self.T_n
        self.pos = self.matrix[:, 1:, 0]
        self.vel = self.matrix[:, 0, 1:]
        self.rel_posvel = self.matrix[:, 1:, 1:]

        self.view = self.central_widget.add_view()
        self.particles = Markers(parent=self.view.scene)
        self.view.add(self.particles)
        self.set_positions()

        self.freeze()

    # =========================================================================================

    def set_positions(self):
        """

        :return:
        :rtype:
        """
        norm_pos = np.zeros((self.N_bods, 3), dtype=np.float64)
        # put a randomized distribution of positions here
        thetas = np.linspace(0, 2 * np.pi, self.N_bods)
        _ph = 0
        mag_pos = np.random.normal(loc=200, scale=50, size=self.N_bods)
        # print("mag_pos =", mag_pos, len(mag_pos))
        norm_pos = np.array(
            [(np.cos(t), np.sin(t), np.sin(_ph)) for t in thetas])  # , dtype=type(np.array(3, dtype=np.float64)))
        mag_pos[0] = 0
        # print("norm_pos =", norm_pos, len(norm_pos))
        self.pos = [mag_pos[n] * norm_pos[n] for n in range(0, self.N_bods)]


    def set_relposvel(self):
        """

        :return:
        :rtype:
        """
        for j, i in [range(0, self.N_bods), range(0, self.N_bods)]:
            # for i in range(0, self.N_bods):
            self.rel_posvel[self.ticks] = self.pos[self.ticks, j] - self.pos[self.ticks, i]


    def set_accel(self):
        """

        :return:
        :rtype:
        """
        for j in range(0, self.N_bods):
            accel = self.zero.copy()
            for i in range(0, self.N_bods):

                if j != i:
                    accel += -self.G * self.mass[i] / np.power(self.rel_posvel[self.T_n, i, j, 2])

            self.matrix[self.ticks, j, j] = accel
            self.accel[self.ticks, j] = accel


    def update_posvel(self):
        """

        :return:
        :rtype:
        """
        self.ticks += 1
        self.vel[self.ticks] += self.accel[self.ticks - 1] * self.warp
        self.pos[self.ticks] += self.vel[self.ticks] * self.warp

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