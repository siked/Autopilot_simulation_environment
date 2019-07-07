import numpy as np


class QuinticPolynomial(object):
    """
    五次多项试，用于求解多项是的微分和导数
    """
   #                 前D位置,d速度,d加速度,0.0, 0.0,时间（4~5,0.2）
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        h = xe - xs;

        self.a0 = xs;
        self.a1 = vxs;
        self.a2 = ( 3 * h - 2 * vxs * T) / (T ** 2);
        self.a3 = (-2 * h + vxs * T) / (T ** 3);

        print(self.a0,self.a1,self.a2,self.a3)


    def calc_point(self, t):
        """
        return the t state based on QuinticPolynomial theory
        """
        xt = self.a0+\
             self.a1*t+\
             self.a2*t**2+\
             self.a3*t**3;
        return xt

    # 速度
    def calc_first_derivative(self, t):
        xt = self.a1 + \
             self.a2*(2*t - 2) + \
             self.a3*3*t**2;
        return xt

    # 加速度
    def calc_second_derivative(self, t):
        xt = self.a2*2 + \
             self.a3*3*(2*t);
        return xt


class QuinticPolynomial_(object):
    """
    五次多项试，用于求解多项是的微分和导数
    """
   #                 前D位置,d速度,d加速度,0.0, 0.0,时间（4~5,0.2）
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs   #当前位置
        self.a1 = vxs  #当前速度
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0] * 1
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        """
        return the t state based on QuinticPolynomial theory
        """
        xt = self.a0 + \
             self.a1 * t + \
             self.a2 * t**2 + \
             self.a3 * t**3 + \
             self.a4 * t**4 + \
             self.a5 * t**5
        return xt

    # below are all derivatives (一阶导数，二阶导数...)
    def calc_first_derivative(self, t):
        xt = self.a1 + \
             2 * self.a2 * t +\
             3 * self.a3 * t**2 + \
             4 * self.a4 * t**3 + \
             5 * self.a5 * t**4
        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + \
             6 * self.a3 * t + \
             12 * self.a4 * t**2 + \
             20 * self.a5 * t**3
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + \
             24 * self.a4 * t + \
             60 * self.a5 * t**2
        return xt

class QuarticPolynomial(object):
    """
    四次多项式
    """

    def __init__(self, xs, vxs, axs, vxe, axe, T):
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4
        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3
        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t
        return xt

class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = 0
        self.ds = []
        self.c = []
import matplotlib.pyplot as plt
def inst():
    lat_qp = QuinticPolynomial_(0, 0, 0, 10, 0.0, 0.0, 8)
    # lat_qp = QuinticPolynomial(w/2     , 0   , 0   , x, 0.0, 0.0, y)
    fp = FrenetPath()
    fp.t = [t for t in np.arange(0.0, 8, 0.1)]
    fp.d = [lat_qp.calc_point(t) for t in fp.t]  # 位移
    fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]  # 速度
    fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]  # 加速度
    # fp.d_ddd = [lat_qp.calc_third_derivative(t) for t input fp.t]  #转向
    # fp.yaw = math.atan2(x - w/2, y)*3  #反正就是要 × 3
    #
    fig, axes = plt.subplots(2, 2)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    # ax4 = axes[1, 1]

    ax1.plot(0, 0)
    ax1.plot(1, 2)
    ax1.plot(fp.d, fp.t, linestyle='--')

    ax2.plot(0, 0)
    ax2.plot(1, 2)
    ax2.plot(fp.d_d, fp.t, linestyle='-')

    ax3.plot(0, 0)
    ax3.plot(1, 2)
    ax3.plot(fp.d_dd, fp.t, linestyle='--')

    plt.show()

inst()
