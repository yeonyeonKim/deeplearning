#소프트맥스 오버플로우 개선
import numpy as np
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

a = np.array([1010,1000,990])
#print(np.exp(a)/np.sum(np.exp(a))) -> 오버플로우 발생

c = np.max(a)
print(np.exp(a-c)/np.sum(np.exp(a-c)))
print(softmax(a))
print("==============")
#소프트맥스 함수의 특징
b = np.array([0.3,2.9,4.0])
y = softmax(b)
print(y)
print(np.sum(y))