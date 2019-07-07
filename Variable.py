Car_L = 1.8  # 车辆轴距，单位：m
Car_Max_Speed  = 3.6 / 3.6  # [m/s]
Car_Acc = 1.2 / 3.6  # [m/s]
Car_Acc_ = 2.0 / 3.6  # [m/s]
Car_Lfc = 1.0  # 前视距离
V_dt = 0.2  # 时间间隔，单位：s
Car_safe = 0.2

# 调节
yaw_k = 7


# 模拟环境
map_H_l = 100   #地图 像素/米
map_W_l = 100   #地图 像素/米

Roi_w = 600  #像素点 总数
Roi_h = 500  #像素点 总数
Roi_w_m = 6  # 单位 米
Roi_h_m = 5  # 单位 米
Roi_w_l = Roi_w/Roi_w_m
Roi_h_l = Roi_h/Roi_h_m


