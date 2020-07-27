#간단한 신경망을 이용한 기울기
import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.function import softmax,cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)#정규분포로 초기화
    def predict(self, x):#예측을 수행
        return np.dot(x,self.W)
    def loss(self,x,t):#손실함수의 값을 구함
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss

def f(W):
    return net.loss(x,t)

net=simpleNet()
x = np.array([0.6,0.9])
t = np.array([0,0,1])
p = net.predict(x)
print(p)
print(np.argmax(p))


print(net.W)
print(net.loss(x,t))
dW = numerical_gradient(f,net.W)
print(dW)