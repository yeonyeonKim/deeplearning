#경사 하강법
import numpy as np
def function(x):
    return x[0]**2+x[1]**2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad
#lr:갱신량
def gradient_descent(f, init_x, lr,step_num):
    x= init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -=lr*grad
    return x

init_x = np.array([-3.0,4.0])
print(gradient_descent(function,init_x,0.1,100))# 0에 가까운 값이므로 정확
init_x = np.array([-3.0,4.0])
print(gradient_descent(function,init_x,10.0,100))#학습률이 너무 크면 큰값으로 발산
init_x = np.array([-3.0,4.0])
print(gradient_descent(function,init_x,1e-10,100))#학습률이 너무 작으면 거의 갱신되지않음