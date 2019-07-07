import math

import cv2
import numpy as np
from Variable import *
from input.Rol import get_Rol
from control.RoBot import Robot
from Analog.Route import get_Route_more
from out.show import show_Route_All,show_Route_r,show_plt
from control.Filter import Filter_paths,Filter_direction
import matplotlib.pyplot as plt
d_x = 300  #小车位置
d_y = 1900  #小车位置
d_yaw = -90  #方向    -90：垂直向上

map = cv2.imread("../620_1.png")
map_ = map.copy()

car = Robot(d_x,d_y,d_yaw)


c_yaw = 0
c_a = 0
# cv2.imshow("map", map)
while 1:


    # if Car_Max_Speed - car.v < Car_Acc:
    #     c_a = 0

    car.move(c_a,c_yaw)
    car_Rol,map_ = Rol = get_Rol(map,map_,car)
    cv2.imshow("Rol", car_Rol)
    # 生成路径
    fp_list = get_Route_more(car.v,Roi_w_m/2,Roi_h_m)
    show_Route_All(car_Rol,fp_list)

    # 过滤 --- 道路
    car_Rol_0 = cv2.split(car_Rol)[0]
    fp_list = Filter_paths(fp_list,car_Rol_0)
    show_Route_All(car_Rol, fp_list)

    # 最后方向筛选
    fp_r = Filter_direction(fp_list,1)
    show_Route_r(car_Rol, fp_r)

    #
    c_yaw = ( np.pi/2 - fp_r.yaw[1] ) * yaw_k
    c_a = fp_r.a[1]
    print("c_yaw:",c_yaw,"c_a:",c_a)

    #show_plt(fp_r)


    pass
    cv2.waitKey(0)


cv2.waitKey(0)
cv2.destroyAllWindows()