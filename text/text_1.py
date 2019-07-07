import math

import cv2
import numpy as np
# from car import Car



class Robot(object):

    def __init__(self, x=160, y=900, yaw=-90, v=20, L=80):
        self.x = x
        self.y = y
        self.yaw = yaw * (np.pi / 180)
        self.yaw_ = yaw * (np.pi / 180)
        self.v = v
        self.L = L
        self.W = 40

    def move(self, a, yaw, dt=0.2):
        sigma = yaw * (np.pi / 180)
        self.x = self.x + self.v * math.cos(self.yaw) * dt
        self.y = self.y + self.v * math.sin(self.yaw) * dt
        self.yaw_ =  self.v / self.L * math.tan(sigma) * dt
        self.yaw += self.yaw_
        self.v = self.v + a * dt
        return self.x, self.y, self.yaw, self.v
    def get_car(self):
        x_a = int(self.x + 10 * math.cos(self.yaw))
        y_a = int(self.y + 10 * math.sin(self.yaw))

        x_b =int( self.x + 10 * math.cos(np.pi/2 + self.yaw))
        y_b =int( self.y + 10 * math.sin(np.pi/2 + self.yaw ))

        x_c = int(self.x + 10 * math.cos(np.pi/2 + self.yaw + np.pi))
        y_c = int(self.y + 10 * math.sin(np.pi/2 + self.yaw + np.pi))

        return x_a,y_a,x_b,y_b,x_c,y_c

    def get_roi(self):
        yaw = 70 * (np.pi / 180)

        v = 20
        x_a = int(self.x + v * math.cos(self.yaw + yaw))
        y_a = int(self.y + v * math.sin(self.yaw + yaw))
        x_b = int(self.x + v * math.cos(self.yaw - yaw))
        y_b = int(self.y + v * math.sin(self.yaw - yaw))
        yaw = 30 * (np.pi / 180)

        v = 200
        x_c = int(self.x + v * math.cos(self.yaw + yaw))
        y_c = int(self.y + v * math.sin(self.yaw + yaw))
        x_d = int(self.x + v * math.cos(self.yaw - yaw))
        y_d = int(self.y + v * math.sin(self.yaw - yaw))

        return x_a,y_a,x_b,y_b ,x_c,y_c,x_d,y_d


    def get_state(self):
        return self.x, self.y, self.yaw, self.v

def xzjz(x,y,x_,y_,angle):
    x_ = x - x_
    y_ = y - y_
    x_ =x + int(x_ * math.cos(angle) - y_ * math.sin(angle))
    y_ =y + int(x_ * math.sin(angle) + y_ * math.cos(angle))
    return x_,y_


d_x = 160
d_y = 900
d_yaw = - 90

car = Robot(d_x,d_y,-90)

map = cv2.imread("zhixian.png")

gray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)

map_z = map.copy()

cv2.circle(map, (d_x, d_y), 3, (0, 255,255), 2)  # 改动最后一个參数
for i in range(40):
    map_ = map.copy()

    w = 0
    av = 0
    if i > 20:
        w = -(i % 20)
        av = -20
    else:
        w = (i % 20)
        av = 20

    car.move(av,w)

    x =  int( car.x)
    y =  int( car.y )
    # cons = cv2.copyMakeBorder(map, x, y, x+40, y+80, cv2.BORDER_CONSTANT, value=0)

    cv2.circle(map_, (x, y), 2, (0, 0, 255), 1)  # 改动最后一个參数
    cv2.circle(map_z, (x, y), 2, (100, 255, 255), 1)  # 改动最后一个參数
    cv2.imshow("map_z", map_z)

    x_a, y_a, x_b, y_b,x_c, y_c = car.get_car()

    cv2.line(map_, (x, y), (x_a, y_a) , (255, 0, 0), 1)
    cv2.line(map_, (x, y), (x_b, y_b), (255, 255, 0), 1)
    cv2.line(map_, (x, y), (x_c, y_c), (255, 255, 0), 1)
    # pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # cv2.polylines(map, [pts], True, (0, 255, 255))

# ROI
    x_a,y_a,x_b,y_b ,x_c,y_c,x_d,y_d = car.get_roi()

    cv2.line(map_, (x, y), (x_c, y_c), (125, 100, 100), 1)
    cv2.line(map_, (x, y), (x_d, y_d), (125, 100, 100), 1)
    cv2.line(map_, (x, y), (x_a, y_a), (250, 100, 100), 1)
    cv2.line(map_, (x, y), (x_b, y_b), (250, 100, 100), 1)

    pts1 = np.float32([[x_d, y_d], [x_b, y_b], [x_a, y_a], [x_c, y_c]])
    # 变换后分别在[A',B',C',D']
    Roi_w = 300
    Roi_h = 600
    pts2 = np.float32([[0,0], [80,Roi_h], [220,Roi_h], [Roi_w,0]])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv2.warpPerspective(map_, M, (Roi_w, Roi_h))
    cv2.imshow("map_", map_)


    cv2.imshow("cons", dst)

    cv2.waitKey(0)



cv2.waitKey(0)
cv2.destroyAllWindows()