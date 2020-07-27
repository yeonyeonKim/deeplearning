#(배치용) 교차 엔트로피 오차구현하기
import numpy as np
#y는 신경망 출력,t는 정답 레이블
def cross_entropy_error1(y,t):
    if y.ndim ==1:#y가 1차원이라면 데이터의 형상을 바꿔준다
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shpe[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size #배치크기로 정규화하고 이미지1장당 평균의 교차엔트로피 오차를 계산

#원-핫 인코딩일때 t가 원소0인 원소는 교차엔트로피의 오차도 0이므로 무시
def cross_entropy_error(y,t):
    if y.ndim ==1:
        t = t.reshape(1,t.size)
        y = t.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.array(batch_size),t]+1e+7))/batch_size