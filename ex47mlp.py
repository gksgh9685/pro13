# MLP란 여러 개의 퍼셉트론 뉴런을 여러 층으로 쌓은 다층신경망 구조.
# 입력층과 출력층 사이에 하나 이상의 은닉층을 가지고 있는 신경망이다.
# 인접한 두 층의 뉴런 간에는 완전 연결 => fully connected 된다.

# 실습1) 논리회로 분류
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
# label = np.array([0,0,0,1])     # and
# label = np.array([0,1,1,1])   # or
label = np.array([0,1,1,0])   # xor 해결함

# max_iter의 추천 횟수 : 500 ~ 1000
ml = MLPClassifier(max_iter=500, hidden_layer_sizes=10,
            solver='adam',   # cost(loss, 손실) 최소화 방식
            learning_rate_init=0.01, verbose=1
).fit(feature, label)

print(ml)
print('입력층 노드 수 : ', ml.n_features_in_)     # 2
print('은닉층 노드 수 : ', ml.hidden_layer_sizes) # 10
print('출력층 노드 수 : ', ml.n_outputs_)         # 1
print('전체층 수 : ', ml.n_layers_)  # 3

# 층 사이 가중치
for i, weight in enumerate(ml.coefs_):
    print(f'가중치 {i} : {weight.shape}')

# 각 층의 편향
for i, bias in enumerate(ml.intercepts_):
    print(f'편향 {i} : {bias.shape}')

pred = ml.predict(feature)
print('pred : ', pred)
print('acc : ', accuracy_score(label, pred))

print('---------')
# 실습2) 일반 자료로 분류
from sklearn.datasets import make_moons  
# 비선형 분류 및 군집화를 위한 초승달 모양 데이터셋 생성 함수
from sklearn.model_selection import train_test_split

x, y = make_moons(n_samples=300, noise=0.2, random_state=42)
print(x[:2])   # [[ 0.80392642 -0.29140734]...
print(y[:2])   # [1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = MLPClassifier(hidden_layer_sizes=(10, 10), solver='adam',
                max_iter=1000, random_state=42, activation='relu')
model.fit(x_train, y_train)

pred = model.predict(x_test)
print('acc : ', accuracy_score(y_test, pred))  # 0.966666
