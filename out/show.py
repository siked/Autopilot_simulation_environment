import cv2
from Variable import *
import matplotlib.pyplot as plt
import numpy as np
# 生成路径
def show_Route_All(car_Rol,fp_list):
    dst = car_Rol.copy()
    for w_list in fp_list:
        cv2.circle(dst, (int((Roi_w / 2) + (w_list.f_x * Roi_w_l)), Roi_h - int(w_list.f_y * Roi_h_l)), 4, (0, 255, 0), 1)
        for w_i in range(len(w_list.x)-2):
            cv2.line(dst,
                     (int((Roi_w/2)+(w_list.x[w_i]*Roi_w_l)),int(Roi_h - w_list.y[w_i]*Roi_h_l)),
                     (int((Roi_w/2)+(w_list.x[w_i+1]*Roi_w_l)),int(Roi_h - w_list.y[w_i+1]*Roi_h_l))
                     , (255, 255, 0), 1)
    cv2.imshow("dst_Route_All", dst)


# 单挑路径
def show_Route_r(car_Rol,fp_r):
    dst = car_Rol.copy()
    cv2.circle(dst, (int((Roi_w / 2) + (fp_r.f_x * Roi_w_l)), Roi_h - int(fp_r.f_y * Roi_h_l)), 4, (0, 255, 0), 1)
    for w_i in range(len(fp_r.x)-2):
        cv2.line(dst,
                 (int((Roi_w/2)+(fp_r.x[w_i]*Roi_w_l)),int(Roi_h - fp_r.y[w_i]*Roi_h_l)),
                 (int((Roi_w/2)+(fp_r.x[w_i+1]*Roi_w_l)),int(Roi_h - fp_r.y[w_i+1]*Roi_h_l))
                 , (255, 255, 0), 1)
    cv2.imshow("show_Route_r", dst)


def show_plt(fp_r):
    fig, axes = plt.subplots(2, 2)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    # ax3 = axes[1, 0]
    # ax4 = axes[1, 1]

    ax1.plot(fp_r.yaw, fp_r.t, ".r", label="course")
    ax1.scatter(np.pi, 0, color="r", s=10)  # 圆点
    ax1.scatter(0, 0, color="r", s=10)  # 圆点


    ax2.plot(fp_r.v, fp_r.t, "-b", label="trajectory")
    # ax3.plot(v_, t, "-r", label="trajectory")

    plt.show()