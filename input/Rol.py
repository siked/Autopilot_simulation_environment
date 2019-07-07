
import cv2
import numpy as np
from Variable import *


# 获取区域
def get_Rol(map,map_,car):

    # car.move(av,w)
    car.get_state()
    x =  int( car.x)
    y =  int( car.y )
    # cons = cv2.copyMakeBorder(map, x, y, x+40, y+80, cv2.BORDER_CONSTANT, value=0)

    cv2.circle(map_, (x, y), 2, (0, 0, 255), 1)  # 改动最后一个參数
    # cv2.circle(map_z, (x, y), 2, (100, 255, 255), 1)  # 改动最后一个參数
    # cv2.imshow("map_z", map_z)

    map_line = map_.copy()
    x_a, y_a, x_b, y_b,x_c, y_c = car.get_car()
    # 水平 方向线
    cv2.line(map_line, (x, y), (x_a, y_a) , (255, 0, 0), 1)
    cv2.line(map_line, (x, y), (x_b, y_b), (255, 255, 0), 1)
    cv2.line(map_line, (x, y), (x_c, y_c), (255, 255, 0), 1)
    # pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # cv2.polylines(map, [pts], True, (0, 255, 255))
    # cv2.imshow("map_", map_)
# ROI
    x_a,y_a,x_b,y_b ,x_c,y_c,x_d,y_d = car.get_roi()
    # 获取 可视范围
    cv2.line(map_line, (x_a, y_a), (x_b, y_b), (255, 255, 100), 1)
    cv2.line(map_line, (x_b, y_b), (x_d, y_d), (255, 255, 100), 1)
    cv2.line(map_line, (x_d, y_d), (x_c, y_c), (250, 255, 100), 1)
    cv2.line(map_line, (x_c, y_c), (x_a, y_a), (250, 255, 100), 1)
    #

    cv2.line(map_line, (x, y), (x_c, y_c), (125, 100, 100), 1)
    cv2.line(map_line, (x, y), (x_d, y_d), (125, 100, 100), 1)
    cv2.line(map_line, (x, y), (x_a, y_a), (250, 100, 100), 1)
    cv2.line(map_line, (x, y), (x_b, y_b), (250, 100, 100), 1)
    cv2.imshow("map_line", map_line)
    pts1 = np.float32([[x_d, y_d], [x_b, y_b], [x_a, y_a], [x_c, y_c]])
    # 变换后分别在[A',B',C',D']
    pts2 = np.float32([[0,0], [240,Roi_h], [360,Roi_h], [Roi_w,0]])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv2.warpPerspective(map, M, (Roi_w, Roi_h))

    return dst,map_

