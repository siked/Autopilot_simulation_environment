import math
from Variable import *
import numpy as np

class Robot(object):

    def __init__(self, x=160, y=900, yaw=-90, v=0):
        self.x = x
        self.y = y
        self.yaw = yaw * (np.pi / 180)
        # self.yaw_ = yaw * (np.pi / 180)
        self.v = v

        self.W = 40

    def move(self, a, yaw):
        self.x = self.x + (self.v * math.cos(self.yaw) * V_dt * map_W_l)
        self.y = self.y + (self.v * math.sin(self.yaw) * V_dt * map_H_l)
        self.yaw +=  self.v / Car_L * math.tan(yaw) * V_dt
        # print(self.yaw*180/np.pi)
        # self.yaw = sigma  -90 * (np.pi / 180)
        self.v = self.v + a * V_dt
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

        v = 0.6*map_W_l
        x_a = int(self.x + v * math.cos(self.yaw + yaw))
        y_a = int(self.y + v * math.sin(self.yaw + yaw))
        x_b = int(self.x + v * math.cos(self.yaw - yaw))
        y_b = int(self.y + v * math.sin(self.yaw - yaw))
        yaw = 30 * (np.pi / 180)

        v = 6*100
        x_c = int(self.x + v * math.cos(self.yaw + yaw))
        y_c = int(self.y + v * math.sin(self.yaw + yaw))
        x_d = int(self.x + v * math.cos(self.yaw - yaw))
        y_d = int(self.y + v * math.sin(self.yaw - yaw))

        return x_a,y_a,x_b,y_b ,x_c,y_c,x_d,y_d


    def get_state(self):
        print("Robot-" ,"v:",self.v,"w:",self.yaw)
        return self.x, self.y, self.yaw, self.v

