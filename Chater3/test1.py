#계단 함수 구현 1
# 넘파이 배열을 인수로 넣을 수 없음
import numpy as np
def step_function1(x):
    if x>0:
        return 1
    else:
        return 0
#계단 함수 구현2
def step_funtion2(x):
    y = x>0
    return y.astype(np.int)

print(step_function1(3.0))
x = np.array([-1.0,1.0,2.0])
print(step_funtion2(x))