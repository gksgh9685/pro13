# sigmoid function 적용 연습
# logisticregression 모델은 wx + b를 logit 변환한 값에 sigmoid function 처리한 값을 사용
# log(P/ (1 - p)) = wx + b라고 정의됨
# 그러므로 z = wx + b --> sigmoid(z) --> p(0 ~ 1) : 이항 분류

# sigmoid function 수식에 의해 변환된 값 확인
import math

def MysigmoidFunc(num):     # num : logit 변환한 값이라 가정
    return 1 / (1 + math.exp(-num))

print(MysigmoidFunc(3))     # 0.9525741268224334
print(MysigmoidFunc(1.5))   # 0.8175744761936437
print(MysigmoidFunc(-5.0))  # 0.0066928509242848
print(MysigmoidFunc(-10))   # 4.5397868702434395

import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

print('\n로짓(logit) 변환된(가정) 값으로 시그모이드 함수 통과 후 결과 시각화')
x = np.linspace(-10, 10, 50)    # 입력자료 (연속형)
print(x)

# 선형결합 (이미 logit된 값)
w = 1.5
b= -2
z = w * x + b

def OurSigmoidFunc(z):      # z : logit 변환한 값이라 가정
    return 1 / (1 + np.exp(-z))

p = OurSigmoidFunc(z)
print('p : \n', p)

# 일부만 보기
print('x[:3] : ', np.round(x[:3], 3))
print('z[:3](logit) : ', np.round(z[:3], 3))
print('p[:3](확률) :', p[:3])

# 시각화
plt.figure(figsize=(8, 5))
plt.plot(x, p, label='sigmoid(z)', color='blue')
plt.axhline(0.5, color='red', linestyle='--')
plt.title('z=wx + b --> sigmoid --> 확률(이형)')
plt.xlabel('x(입력값)')
plt.ylabel('y(입력값)')
plt.grid(True)
plt.legend()
plt.show()

# 로지스틱 회귀분석은 독립변수와 범주형 종속변수의 관계를 학습하여
# 분류를 수행하는 알고리즘이다.
# 독립변수에 학습된 가중치와 절편을 적용하여 z = wx + b를 계산한다.
# 이 값은 확률 자체가 아니라 로그 오즈(Logit) 값으로,
# -무한대부터 +무한대까지의 실수 범위를 가진다.
# 이후 시그모이드 함수를 적용하여 0과 1 사이의 확률을 구하고,
# 일반적으로 0.5를 기준으로 클래스를 분류한다.

# X → wx+b(이미 로짓 값) → 시그모이드 → 확률 → 클래스 분류
# 로지스틱 회귀는 wx + b를 계산하기 전에 선형회귀의 추세선을 별도로 학습하는 것이 아니라
# 학습 과정에서 분류에 적합한 가중치w와 절편b를 직접 추정한다.


