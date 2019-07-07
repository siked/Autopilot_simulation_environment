import math

import cv2
import numpy as np

from text_2_frenet import get_list
# from car import Car

ROBOT_RADIUS = 10  # robot radius [m]
Roi_w = 300  #像素点 总数
Roi_h = 600  #像素点 总数
Roi_w_m = 3  # 单位 米
Roi_h_m = 6  # 单位 米
Roi_w_l = Roi_w/Roi_w_m
Roi_h_l = Roi_h/Roi_h_m
class Robot(object):

    def __init__(self, x=160, y=900, yaw=-90, v=0, L=180):
        self.x = x
        self.y = y
        self.yaw = yaw * (np.pi / 180)
        self.yaw_ = yaw * (np.pi / 180)
        self.v = v
        self.L = L
        self.W = 40

    def move(self, a, yaw, dt=0.1):
        self.x = self.x + self.v * math.cos(self.yaw) * dt
        self.y = self.y + self.v * math.sin(self.yaw) * dt
        self.yaw +=  self.v / self.L * math.tan(yaw) * dt
        # print(self.yaw*180/np.pi)
        # self.yaw = sigma  -90 * (np.pi / 180)
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
        print("Robot-" ,"s:",self.v,"w:",self.yaw)
        return self.x, self.y, self.yaw, self.v




# 当前坐标是不是在安全范围

def check_collision(fp, ob):
    for x in range(len( fp.y)-1):
        ob_y = int(600 - (fp.y[x+1] * Roi_h_l))
        ob_x = int((Roi_w/2)+(fp.x[x+1] * Roi_w_l))

        a_x = ob_x + ROBOT_RADIUS
        a_y = ob_y + ROBOT_RADIUS

        b_x = ob_x - ROBOT_RADIUS
        b_y = ob_y - ROBOT_RADIUS


        if a_x >= 300 : a_x = 300-1
        if b_x >= 300 : b_x = 300-1

        if a_y >= 600 : a_y = 600-1
        if b_y >= 600 : b_y = 600-1

        if a_x < 1 : a_x = 0
        if b_x < 1 : b_x = 0

        if a_y < 1 : a_y = 0
        if b_y < 1 : b_y = 0

        if ob_y >= 600 : ob_y = 600-1
        if ob_x >= 600 : ob_x = 600-1

        if ob_y < 1 : ob_y = 0
        if ob_x < 1 : ob_x = 0

        ob_i = ob[ob_y,ob_x]

        ob_i += ob[a_y, a_x]
        ob_i += ob[a_y, b_x]
        ob_i += ob[b_y, a_x]
        ob_i += ob[b_y, b_x]

        # ob_i = ob[a_x:a_y, b_x:b_y].all()
        if ob_i > 0 :
            return False



    return True


# #第一层过滤
# MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
# MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
# MAX_CURVATURE = 1.0  # maximum curvature [1/m]
def check_paths(fplist, ob):
    okind = []
    kind = []
    for i in range(len(fplist)):
        # if any([v > MAX_SPEED for v input fplist[i].s_d]):  # Max speed check
        #     continue
        # elif any([abs(a) > MAX_ACCEL for a input fplist[i].s_dd]):  # Max accel check
        #     continue
        # elif any([abs(c) > MAX_CURVATURE for c input fplist[i].c]):  # Max curvature check
        #     continue
        if not check_collision(fplist[i], ob):
            kind.append(fplist[i])
            continue
        okind.append(fplist[i])
    return okind,kind


#最后方向筛选

def check_direction(fplist,direction):
    dt_x = 1
    dt_y = 1
    if direction == 0:
        dt_x = -1
        dt_y = 0.5
    elif direction == 1:
        dt_x = 1
        dt_y = 1
    elif direction == 2:
        dt_x = 1
        dt_y = 0.5

    for fp in fplist:
        fp.cf  = ((1/Roi_w_m)*((Roi_w/Roi_w_l)/2 + fp.f_x)*dt_x) + (fp.f_y*dt_y)
    return fplist
def as_num(x):
    y='{:.7f}'.format(x)
    return(y)

d_x = 160
d_y = 1000
d_yaw = - 90

car = Robot(d_x,d_y,-90)

map = cv2.imread("zhixian_1.png")

# gray = cv2.cvtColor(map, cv2.Col,1)

map_z = map.copy()

cv2.circle(map, (d_x, d_y), 3, (0, 255,255), 2)  # 改动最后一个參数
w = 0
av = 0
while 1:
    map_ = map.copy()


    car.move(av,w)
    car.get_state()
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
    cv2.imshow("map_", map_)
    pts1 = np.float32([[x_d, y_d], [x_b, y_b], [x_a, y_a], [x_c, y_c]])
    # 变换后分别在[A',B',C',D']

    pts2 = np.float32([[0,0], [80,Roi_h], [220,Roi_h], [Roi_w,0]])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv2.warpPerspective(map, M, (Roi_w, Roi_h))
    cv2.imshow("dst", dst)




# 生成路径
    fp_list = get_list(car.v,(Roi_w/Roi_w_l)/2,Roi_h/Roi_h_l)
    dst_1 = dst.copy()
    for w_list in fp_list:
        for w_i in range(len(w_list.x)-2):
            cv2.line(dst_1,
                     (int((Roi_w/2)+(w_list.x[w_i]*Roi_w_l)),int(Roi_h - w_list.y[w_i]*Roi_h_l)),
                     (int((Roi_w/2)+(w_list.x[w_i+1]*Roi_w_l)),int(Roi_h - w_list.y[w_i+1]*Roi_h_l))
                     , (255, 255, 0), 1)
    cv2.imshow("dst_1", dst_1)

    #过滤 --- 道路
    b = cv2.split(dst)[0]
    fp_list,k_fp_list = check_paths(fp_list, b)

    dst_2 = dst.copy()
    for w_list in fp_list:
        for w_i in range(len(w_list.x) - 2):
            cv2.line(dst_2,
                     (int((Roi_w/2)+(w_list.x[w_i]*Roi_w_l)), int(Roi_h - w_list.y[w_i]*Roi_h_l)),
                     (int((Roi_w/2)+(w_list.x[w_i + 1]*Roi_w_l)), int(Roi_h - w_list.y[w_i + 1]*Roi_h_l))
                     , (255, 255, 0), 1)
    cv2.imshow("dst_2", dst_2)

    #过滤 --- 道路
    b = cv2.split(dst)[0]

    fp_list = check_direction(fp_list, 1)


    if len(fp_list) == 0 : break  #跳出循环


    dst_3 = dst.copy()
    cf_max = 0

    cc_fp = fp_list[1]
    for w_list in fp_list:
        if cf_max < w_list.cf:
            cf_max = w_list.cf
            cc_fp = w_list
        cv2.circle(dst_3, (int((Roi_w/2)+(w_list.f_x*Roi_w_l)), 600 - int(w_list.f_y*Roi_h_l)), 2, (255-int(w_list.cf), 255-int(w_list.cf), 255), 1)  # 改动最后一个參数

    for w_i in range(len(cc_fp.y) - 2):
        cv2.line(dst_3,
                 (int((Roi_w/2)+(cc_fp.x[w_i]*Roi_w_l)), int(Roi_h - cc_fp.y[w_i]*Roi_h_l)),
                 (int((Roi_w/2)+(cc_fp.x[w_i + 1]*Roi_w_l)), int(Roi_h - cc_fp.y[w_i + 1]*Roi_h_l))
                 , (255, 255, 0), 1)
    w = cc_fp.yaw[1]
    av = cc_fp.v[1]
    cv2.circle(dst_3, (int((Roi_w/2)+(cc_fp.f_x*Roi_w_l)),600 - int(cc_fp.f_y*Roi_h_l)), 4, (0, 255, 0), 1)

    cv2.imshow("dst_3", dst_3)

    #print("fp.cf=",cc_fp.cf)
    #print("w=", w)

    cv2.imshow("b", b)


    pass
    cv2.waitKey(0)
#


cv2.waitKey(0)
cv2.destroyAllWindows()