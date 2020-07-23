#perceptron test 'NAND gate and OR gate'
import numpy as np
def NADN(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(x*w)+b
    if tmp <=0:
        return 0
    elif tmp>0:
        return 1
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.4
    tmp = np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1

print(NADN(1.0,1.0))
print(OR(0.5,0.5))