import matplotlib.pyplot as plt
import numpy as np
import sympy
t = sympy.symbols('t')
y=3*pow(t,4)-4*pow(t,3)-12*pow(t,2)
y_diff1 = sympy.diff(y, t)
y_diff2 = sympy.diff(y, t, 2)
f=sympy.lambdify(t,y,"numpy")
f1=sympy.lambdify(t,y_diff1,"numpy")
f2=sympy.lambdify(t,y_diff2,"numpy")
# print(sympy.solveset(sympy.Eq(y_diff1,0)))
station=sympy.solveset(sympy.Eq(y_diff1,0))
station=list(station)
station=list(map(int,station))
# 去除鞍点
for i,a in enumerate(station):
    if f2(np.array(a))==[0]:
        station=station.remove(a)
    pass
print("所有驻点",station)
station.sort()
region=0.2*(station[-1]-station[0])
t=np.arange(station[0]-region,station[-1]+region,0.01)
fg,axes=plt.subplots(3)
axes[0].plot(t,f(t))
axes[0].set_title("original function")
axes[1].plot(t,f1(t))
axes[1].plot(station,f1(np.array(station)),'ro')
# axes[1].set_title("一阶导数")
axes[2].plot(t,f2(t))
# axes[2].set_title("二阶导数")

for i in range(3):
    axes[i].grid()
    pass
plt.show()