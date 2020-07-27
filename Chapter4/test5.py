import numpy as np
import matplotlib.pylab as plt
#미분의 나쁜예 ->반올림 오차가 생김
def numberical_diff1(f,x):
    h = 10e-50
    return (f(x+h)-f(x))/h
#중앙차분 중심차분으로 오차를 줄임(x를 중심으로 그 전,후를 구함)
def numberical_diff2(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

def function(x):
    return 0.01*x**2+0.1*x

def tangent_line(f,x):
    d = numberical_diff2(f,x)
    print(d)
    y = f(x)-d*x
    return lambda t: d*t+y

x = np.arange(0.0,20.0,0.1)
y = function(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function,5)
y2 = tf(x)
plt.plot(x,y)
plt.plot(x,y2)
plt.show()

