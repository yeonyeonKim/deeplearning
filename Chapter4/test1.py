#오차제곱의 합
import numpy as np
def sum_square_error(y,t):
    return 0.5*np.sum((y-t)**2)

t = np.array([0,0,1,0,0,0,0,0,0,0])

y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
print(sum_square_error(y,t))
y = np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])
print(sum_square_error(y,t))