#ReLU함수
import numpy as np
def relu(x):
    return np.maximum(0,x)

x = np.array([-0.1,-0.3,3,4])
print(relu(x))