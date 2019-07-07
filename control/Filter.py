from Variable import *


# 当前坐标是不是在安全范围
def Filter_collision(fp, ob):
    for x in range(len(fp.x)-2):
        ob_y = int(Roi_h - (fp.y[x+1] * Roi_h_l))
        ob_x = int((Roi_w/2)+(fp.x[x+1] * Roi_w_l))

        if ob_y >= Roi_h : ob_y = Roi_h - 1
        if ob_x >= Roi_w : ob_x = Roi_w - 1

        if ob_y < 1 : ob_y = 0
        if ob_x < 1 : ob_x = 0

        a_x =int( ob_x + Car_safe * Roi_w_l)
        a_y =int( ob_y + Car_safe * Roi_h_l)

        b_x =int( ob_x - Car_safe * Roi_w_l)
        b_y =int( ob_y - Car_safe * Roi_h_l)


        if a_x >= Roi_w : a_x = Roi_w - 1
        if b_x >= Roi_w : b_x = Roi_w - 1

        if a_y >= Roi_h : a_y = Roi_h - 1
        if b_y >= Roi_h : b_y = Roi_h - 1

        if a_x < 1 : a_x = 1
        if b_x < 1 : b_x = 1

        if a_y < 1 : a_y = 1
        if b_y < 1 : b_y = 1



        a_x =int(a_x)
        a_y =int(a_y)
        b_x =int(b_x)
        b_y =int(b_y)

        # print(a_x,a_y,b_x,b_y)

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
def Filter_paths(fplist, ob):
    okind = []
    for i in range(len(fplist)):
        # if any([v > MAX_SPEED for v input fplist[i].s_d]):  # Max speed check
        #     continue
        # elif any([abs(a) > MAX_ACCEL for a input fplist[i].s_dd]):  # Max accel check
        #     continue
        # elif any([abs(c) > MAX_CURVATURE for c input fplist[i].c]):  # Max curvature check
        #     continue
        if not Filter_collision(fplist[i], ob):
            continue
        okind.append(fplist[i])
    return okind



# 最后方向筛选
def Filter_direction(fplist,direction):
    fp_r = 0
    fp_cf = 0
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
        if fp_cf < fp.cf:
            fp_cf = fp.cf
            fp_r = fp

    return fp_r


