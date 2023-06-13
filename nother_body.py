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
        self.T_n = np.linspace(start=self.T_0,
                               stop=self.T_1,
                               num=self.N_frames)
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
                                3), dtypre=np.float64)

        self.frame_t = self.matrix.[:, 0, 0]
        self.frame_t = self.T_n
        self.pos = self.matrix[:, 1:, 0]
        self.vel = self.matrix[:, 0, 1:]
        self.rel_posvel = self.matrix[:, 1:, 1:]

        self.view = self.central_widget.add_view()
        self.view.camera = "arcball"
        self.freeze()

    # ==============================================================================

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
                    accel += -self.G * self.mass[i] / np.power(self.rel_posvel[self.T_n, 2])

            self.matrix[self.ticks, j, j] = accel


    def update_posvel(self):
        """

        :return:
        :rtype:
        """
        self.vel[]