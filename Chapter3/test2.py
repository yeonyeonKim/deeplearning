#계단 함수 그래프
import numpy as np
import matplotlib.pylab as plt

def step_funtion(x):
    return np.array(x>0,dtype=np.int)

x = np.arange(-5.0,5.0,0.1)
y = step_funtion(x)
plt.plot(x,y)#x,y배열로 그래프를 그린다
plt.ylim(-0.1,1.1)#y축 범위 지정
plt.show()