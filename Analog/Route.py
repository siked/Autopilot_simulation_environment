import numpy as np
import time
from Variable import *



import numpy as np
import math
import matplotlib.pyplot as plt
import time
# k = 1.0  # 前视距离系数
# Lfc = 1.0  # 前视距离
Kp = 0.1  # 速度P控制器系数

# Ap_ = 1.0  #减速度系数

#   单条路径生成

class Route_one:

    def __init__(self, x=0.0, y=0.0, v=0.0, yaw=np.pi/2,f_x=0.0, f_y=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.f_x = f_x
        self.f_y = f_y

    def update(self, a_v, a_yaw):

        self.x = self.x + self.v * math.cos(self.yaw) * V_dt
        self.y = self.y + self.v * math.sin(self.yaw) * V_dt
        self.yaw = self.yaw + (self.v / Car_L * math.tan(a_yaw) * V_dt)
        self.v = self.v + a_v * V_dt

    def pure_pursuit_control(self):
        alpha = (math.atan2(self.f_y - self.y, self.f_x - self.x) - self.yaw)
        Lf = Kp * self.v + Car_Lfc
        delta = math.atan2(2.0 * V_dt * math.sin(alpha) / Lf, 1.0)
        return alpha,alpha


    #计算 减速度
    def ac_(self,t,v,a):
        av = 0 #初始化
        for t_x in range(len(t)):
            t_x = len(t) - t_x - 1
            a_v = (v[t_x] - av)
            if a_v < 0:
                break
            av = av + Car_Acc_ * V_dt
            v[t_x] = av
            a[t_x] = -a_v
        return v,a

    def go(self):


        T = 100.0  # 最大模拟时间

        # 设置车辆的出事状态
        # lastIndex = len(cx) - 1
        time = 0.0
        x = [self.x]
        y = [self.y]
        yaw = [self.yaw]
        v = [self.v]
        t = [0.0]

        ac = [0]

        while T >= time:


            di_, alpha = self.pure_pursuit_control()

            ai = Car_Max_Speed - self.v

            if_y = abs(self.f_y) - abs(self.y)
            if_x = abs(self.f_x) - abs(self.x)
            if if_y < -0.1 or if_x < -0.1:
                break;

            self.update( ai, di_)

            time = time + V_dt

            x.append(self.x)
            y.append(self.y)
            yaw.append(self.yaw)
            v.append(self.v)
            t.append(time)
            ac.append(ai)

        v,a = self.ac_(t,v,ac)

        if abs(if_x) > 0.1 :
            return 0, t, x, y, yaw, v, ac


        return 1,t,x,y,yaw,v,ac


def main_Route_one():
    #  设置目标路点
    cx = -3
    cy = 10

    VS = Route_one(x=-0.0, y=0.0, yaw=np.pi / 2, v=0.3/3.6,f_x= cx,f_y = cy)
    _,t,x,y,yaw,v,ac = VS.go()

    fig, axes = plt.subplots(2, 2)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    ax1.plot(cx, cy, ".r", label="course")
    ax1.plot(x, y, "-b", label="trajectory")

    ax2.plot(yaw, t, ".r", label="course")
    ax2.scatter(np.pi, 0, color="r", s=10)  # 圆点
    ax2.scatter(0, 0, color="r", s=10)  # 圆点
    ax2.plot([np.pi/2,np.pi/2], [0, t[len(t)-1]])  # 线   -线  .点


    ax3.plot(v, t, "-b", label="trajectory")
    # ax3.plot(v_, t, "-r", label="trajectory")

    ax4.plot(ac, t, "-b", label="trajectory")
    # ax4.plot(a_, t, "-r", label="trajectory")

    plt.show()
#测试单挑路径
#main_Route_one()



# 生成 所有路径

class FPath:

    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.a = []
        self.f_x = 0
        self.f_y = 0
        self.cf = 0

def get_Route_more(i_v,w,h):
    is_show = float #
    if is_show:time_a = time.time()
    fp_ = []
    for f_x in np.arange(-w+0.2, w, 0.2):
        for f_y in np.arange(1, h+1, 1):

            if is_show: plt.scatter(f_x, f_y, color="r", s=10)  # 圆点
            VS = Route_one(x=0, y=0.0, yaw=np.pi / 2, v=i_v ,f_x=f_x, f_y=f_y)
            _,t_, x_, y_, yaw_, v_, ac_ = VS.go()
            if _ == 0:
                continue;
            fp = FPath()
            if is_show: plt.plot(x_, y_, linestyle='--')
            fp.t=t_
            fp.x=x_
            fp.y=y_
            fp.yaw=yaw_
            fp.v=v_
            fp.a=ac_
            fp.f_x = f_x
            fp.f_y = f_y
            fp_.append(fp)
    # plt.show()
    print(time.time() - time_a)
    return fp_


#get_Route_more(0.1,1.5,10)















