# Perceptron :sklearn이 제공하는 단층신경망(뉴런, 노드)
# 기본적으로 이항분류(선형, wx + b) 가능 
# 활성화 함수는 기본적으로 계단함수(step function)를 사용
# 퍼셉트론은 선형 결합 → 활성화 함수 → 클래스 예측 → 오분류 시 가중치 갱신 순서로 동작

# 실습1) 논리회로 분류
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
# label = np.array([0,0,0,1])   # and
# label = np.array([0,1,1,1])   # or
label = np.array([0,1,1,0])   # xor 해결 못함 ()

ml = Perceptron(max_iter=100).fit(feature, label)  # max_iter=학습횟수. 1회학습은 epoch
print(ml)
pred = ml.predict(feature)
print('pred : ', pred)
print('acc : ', accuracy_score(label, pred))

# Perceptron은 딥러닝의 경사하강법과는 달리 틀린 것만 고치는 알고리즘
# 흐름 : 예측 -> 맞았는지 확인 -> 틀리면 Weight를 갱신, 맞으면 통과 -> 이 과정 반복
# 선형회귀식 사용(LogisticRegresion에서 힌트 얻음) : 선형 결합으로 결정 경계 계산
#  - input에 대한 가중치 합 계산 후 실제값과 예측값 비교(Loss function)
#  - 기본 퍼셉트론 학습 규칙으로 직접 가중치(W) 갱신하기를 max_iter 만큼 반복함

# 학습 원리
# 1. 입력값과 가중치의 선형 결합 계산 (z = wx + b)
# 2. 활성화 함수(계단 함수)를 사용하여 선형결합의 결과를 0 또는 1로 변환
#    계산 결과를 이용해 클래스 예측
# 3. 실제값과 예측값 비교
# 4. 틀리게 예측한 경우 가중치(W)와 편향(b)을 갱신
# 5. 전체 학습 데이터를 반복하면서 분류 오류를 줄임

print('---------')
# 실습2) 일반 자료 분류
x = np.array([
    [2,3],
    [3,3],
    [1,1],
    [5,2],
    [6,1]
])
y = np.array([1,1,1,-1,-1])

model2 = Perceptron(max_iter=100, eta0=0.1, random_state=42)  # eta0 : 학습률
model2.fit(x, y)
pred2 = model2.predict(x)
print('예측값 : ', pred2)
print('실제값 : ', y)
print('정확도 : ', accuracy_score(y, pred2))
# parameter 확인
print('가중치(w, weight) : ', model2.coef_)   # [[-0.4  0.8]]
print('편향(b, bias) : ', model2.intercept_)  # [0.4]

# 결정 경계 (w1*x1 + w2*x2 + b) 시각화
import matplotlib.pyplot as plt
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr')
w = model2.coef_[0] 
b = model2.intercept_[0]
x_vals = np.linspace(0, 7, 100)
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals)
plt.title('sklearn Perceptron')
plt.xlabel('x1')
plt.xlabel('x2')
plt.show()