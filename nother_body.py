import logging
import numpy as np
from numpy import ndarray
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

    def __init__(self,
                 num_bods=7,
                 t_start=1, t_end=100,
                 n_frames=20,
                 has_star=True,
                 star_mass=10000,
                 *args, **kwargs):
        """

        :type T0: np.float64
        :param num_bods:
        """
        global norm_pos, mag_pos
        super(NewtonMatrix, self).__init__(title="Figuring out Markers, Numpy, Transforms and SHIT",
                                           keys='interactive',
                                           *args, **kwargs)
        self.unfreeze()
        G_base = 6.67430e-11                        # gravitational constant
        G_tweak = 1.0                               # tweak gravity by this factor
        self.G = G_base * G_tweak                   #
        self.N_frames = n_frames                    # number of snapshots over time interval
        self.ticks = 0
        self.warp = 0.01                            # 'time' elapsed per frame
        self.has_star = has_star
        self.star_mass = star_mass
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
        self.frame_t = np.linspace(start=t_start, stop=t_end, num=n_frames)
        self.pos = self.matrix[:, 1:, 0]
        self.vel = self.matrix[:, 0, 1:]
        self.rel_posvel = self.matrix[:, 1:, 1:]

        self.view = self.central_widget.add_view()
        self.axes = axis_visual(scale=10000, parent=self.view.scene)
        self.particles = Markers(parent=self.view.scene)
        self.init_pos0()
        self.view.add(self.axes)
        self.view.add(self.particles)
        self.timer = app.Timer(interval=self.warp, connect=self.iterate, start=False, app=self.app)
        self.freeze()

    # =========================================================================================

    def init_pos0(self):
        """

        :return:
        :rtype:
        """
        # put a randomized distribution of positions here
        self.mass = np.linspace(start=10, stop=11, num=self.N_bods)
        th = np.linspace(0, 2 * np.pi, self.N_bods)
        cos_th = np.cos(th)
        sin_th = np.sin(th)
        mag_pos = np.random.normal(loc=100, scale=20, size=self.N_bods)
        _ph = np.zeros(self.N_bods, dtype=np.float64)

        if self.has_star:
            mag_pos[0] = 0
            self.mass[0] = self.star_mass
        pos0 = np.array([cos_th, sin_th, _ph]).reshape(3, self.N_bods)
        print(mag_pos.shape, _ph.shape, pos0.shape)
        self.pos = mag_pos * pos0
        self.iterate(None)

    def set_rel_pv(self):
        """
        """
        for j in range(0, self.N_bods):
            for i in range(0, self.N_bods):
                self.rel_posvel[:, i, j] = self.pos[:, j] - self.pos[:, i]

    def set_accel(self):
        """
        """
        for j in range(0, self.N_bods):
            accel = self.zero.copy()
            for i in range(0, self.N_bods):

                if j != i:
                    d_sqr = np.linalg.norm(self.rel_posvel[self.ticks, i, j])
                    d_sqr *= d_sqr
                    if d_sqr != 0:
                        accel += -self.G * self.mass[i] / d_sqr

            self.matrix[self.ticks, j, j] = accel
            self.accel[self.ticks, j] = accel
            d_accel = np.linalg.norm(accel - self.accel[self.ticks - 1, j])
            d_d_accel = d_accel - np.linalg.norm(self.matrix[self.ticks - 1, j, j] - self.matrix[self.ticks - 2, j, j])
            # print("accel[", j, "] =", accel)
            # print("\td_d_accel[", j, "] =", d_d_accel, "\n")

    def update_posvel(self):
        """
        """
        for i in range(0,self.N_bods - 1):
            # print("position[", i, "] =\n", self.pos[self.ticks], self.pos[self.ticks].shape)
            # print("velocity[", i, "] =\n", self.vel[self.ticks], self.vel[self.ticks].shape)
            self.vel[self.ticks + 1] = self.vel[self.ticks] + self.matrix[self.ticks, i + 1, i + 1] * self.warp
        print(self.pos[self.ticks + 1], "\n", self.pos[self.ticks], "\n",  self.vel[self.ticks + 1], "\n", self.warp)
        self.pos[self.ticks + 1] = self.pos[self.ticks] + self.vel[self.ticks + 1] * self.warp

        self.ticks += 1
        self.ticks = self.ticks % self.N_frames - 1

    def iterate(self, event):
        """     This method is called by the timer.
            It computes the relative positions, the resulting accelerations, then updates the next pos and vel.
            Finally, is sets the pos data for the particle markers.
        :return:
        :rtype:
        """
        self.set_rel_pv()
        self.set_accel()
        self.update_posvel()
        print(self.pos[self.ticks])
        self.particles.set_data(pos=self.pos[self.ticks], size=2, edge_color="red", edge_width=1)
        logging.info(str(self.matrix[self.ticks]))


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
    # print("MAIN")

    trx = MatrixTransform()
    trx.rotate(-90, [1, 0, 0])
    trx.scale((1, 1))

    can = NewtonMatrix(fullscreen=False)
    can.init_pos0()
    can.axes.transform = trx
    can.axes.visible = True
    can.particles.visible = True
    can.show()
    can.timer.start()
    can.app.run()


if __name__ == '__main__':
    main()