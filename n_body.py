import numpy as np
from vispy import app
from vispy.scene import SceneCanvas
from vispy.scene.visuals import XYZAxis, Markers
from vispy.visuals.transforms import MatrixTransform
from vispy.scene.cameras import FlyCamera


class NewtonMatrix(Markers):
    """

    """
    def __init__(self, num_bods=10, mass=[1.0, ] * 10, pos_0=None, vel_0=None, *args, **kwargs):
        """

        :type T0: np.float64
        :param num_bods:
        """
        super(NewtonMatrix, self).__init__(*args, **kwargs)
        self.unfreeze()
        self.zero = np.zeros((1, 3), dtype=np.float64)
        print("ZERO =", self.zero)
        self.G = 6.67430e-11
        self.T0 = self.zero
        self.T_MAX = 1000
        self.N_BODS = num_bods

        if mass is None:
            self.mass = np.ndarray((self.N_BODS, 1), dtype=np.float64)
            # put a randomized distribution of masses here
            self.mass = np.random.normal(loc=1000, scale=100, size=self.N_BODS)
            print("MASS =", self.mass)
        else:
            self.mass = mass

        if pos_0 is None:
            self.pos_0 = np.zeros((self.N_BODS, 1), dtype=type(self.zero))
            # put a randomized distribution of positions here
            _th = np.linspace(0, 2 * np.pi, self.N_BODS)
            _r = np.random.normal(loc=1000, scale=100, size=self.N_BODS)
            pos_0 = np.array([_r * np.cos(_th), _r * np.sin(_th)])
        else:
            self.pos_0 = pos_0

        if vel_0 is None:
            self.vel_0 = np.zeros((self.N_BODS, 1), dtype=type(self.zero))
            # put a randomized distribution of velocities here
            dv = np.random.normal(loc=0, scale=15 * np.pi / 180, size=self.N_BODS)
            self.vel_0 = np.cross([0, 0, 1], self.pos_0)
        else:
            self.vel_0 = vel_0

        assert (len(self.mass) == self.N_BODS)
        assert (len(self.pos_0) == self.N_BODS)
        assert (len(self.vel_0) == self.N_BODS)

        [print(self.mass[n], self.pos_0[n], self.vel_0[n]) for n in range(0, self.N_BODS - 1)]
        self.SM = np.ndarray((self.N_BODS + 1, self.N_BODS + 1, 3), dtype=np.float64)
        print(type(self.SM), "\n")
        self.SM[0, 0] = self.T0
        self.SM[1:, 0] = pos_0
        self.SM[0, 1:] = vel_0
        # print(type(self.SM[1, 0]), type(self.SM[0, 1]))

        self.freeze()
        self.set_rel_posvel()
        self.set_accel()

    def set_rel_posvel(self):
        """
        :return:
        """
        for i in range(1, self.N_BODS + 1):
            for j in range(1, self.N_BODS + 1):
                if i == j:
                    print("(i=j): ZERO???", self.SM[j, 0], self.SM[i, 0])
                    self.SM[i, j] = self.zero
                elif i > j:
                    print("(i<j): ", self.SM[j, 0], self.SM[i, 0])
                    self.SM[i, j] = self.SM[j, 0] - self.SM[i, 0]
                elif i < j:
                    print("(i>j): ", self.SM[0, j], self.SM[0, i])
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
                    print(i, j, ":dist_sqr (i>j) = ", dist_sqr)
                    if dist_sqr != 0:
                        accel += self.SM[i, j] * (-self.G * self.mass[j] / dist_sqr)
                elif i < j:
                    dist_sqr = np.linalg.norm(-self.SM[i, j], axis=0)
                    dist_sqr *= dist_sqr
                    print(i, j, ":dist_sqr (i<j) = ", dist_sqr)
                    if dist_sqr != 0:
                        accel += -self.SM[i, j] * (-self.G * self.mass[j] / dist_sqr)
                elif i == j:
                    dist_sqr = 0.0
                    print(i, j, ":dist_sqr (i=j) = ", dist_sqr)

            self.SM[j, j] = accel
        [print("accel[", n, "] =", self.SM[n, n]) for n in range(1, self.N_BODS)]

    def iterate(self):
        """ Use acceleration values to update vel, then pos."""
        return self.SM


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


def not_main():
    print("MAIN")
    can = SceneCanvas(title="Testing Transforms", size=(800, 600), keys="interactive", show=True)
    view = can.central_widget.add_view()
    view.camera = FlyCamera()
    view.camera.set_range()
    viz = axis_visual(scale=3, parent=view.scene)
    view.add(viz)
    can.show()
    app.run()


def main():
    N = 10
    MAX_T = 10
    T = 0
    mass = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., ]
    theta = np.linspace(0, 359, N)
    radius = np.random.normal(loc=1000, scale=100, size=N)
    pos_0 = np.zeros((N, 3), dtype=np.float64)
    vel_0 = np.zeros((N, 3), dtype=np.float64)
    vel0_dir = np.zeros((N, 3), dtype=np.float64)

    for n in range(0, N - 1):
        r = radius[n]
        ct = np.cos(theta[n])
        st = np.sin(theta[n])
        print(r, ct, st)
        pos_0[n] = [r * ct, r * st, 0]
        vel0_dir[n] = np.cross(pos_0[n], [0.0, 0.0, 1.0]) / np.linalg.norm(pos_0[n])
        vel_0[n] = vel0_dir[n] * (1 / np.sqrt(np.linalg.norm(pos_0[n])))

    NM_0 = NewtonMatrix(mass=mass, pos_0=pos_0, vel_0=vel_0)
    NM_T = [NM_0.SM]

    while T < MAX_T:
        NM_1 = NM_0.iterate()
        NM_T.append(NM_1)
        NM_0 = NM_1.copy()
        T += 1
    print(len(NM_T))


if __name__ == '__main__':
    main()
