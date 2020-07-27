#기울기
import numpy as np
def function(x):
    return x[0]**2+x[1]**2
def numerical_gradient(f,x):
    h=1e-4
    grad = np.zeros_like(x)#x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val =x[idx]
        #f(x+h)
        x[idx] = tmp_val+h
        fxh1 = f(x)
        #f(x-h)
        x[idx] = tmp_val-h
        fxh2 = f(x)
        
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val

    return grad

print(numerical_gradient(function,np.array([3.0,4.0])))
print(numerical_gradient(function,np.array([0.0,2.0])))
print(numerical_gradient(function,np.array([3.0,0.0])))