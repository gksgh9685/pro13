# 다층 인공신경망(MLP)으로 다항 분리 (3개)
# sklearn.datasets : wine
# 총 샘플 수: 178개
# 클래스(타깃) 수: 3개 클래스 (각각 59개, 71개, 48개 샘플로 구성)
# 특성(Features) 수: 13개의 실수형(Real-valued) 및 양수(Positive) 특성


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_wine()
x = data.data
y = data.target
print(x[:2], ' ', x.shape)     # (178, 13)
print(y[:2], ' ', np.unique(y))  # [0 0]   [0 1 2]

x_train,x_test,y_train,y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Scaleing (MLP에서 권장)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# 모델 생성
model = MLPClassifier(
    hidden_layer_sizes=(20, 10),   # 은닉층 2개(첫번째:노드 20, 두번째: 노드 10개)
    solver='adam',     # 손실 최소화
    activation='relu', # 은닉층의 활성화 함수  
    max_iter=150,
    learning_rate_init=0.001,
    random_state=42,
    verbose=1   # 학습 도중 로그 출력 여부
)

model.fit(x_train_scaled, y_train)
pred = model.predict(x_test_scaled)
print('accuracy_score : ', accuracy_score(y_test, pred))  # 0.91666
print(classification_report(y_test, pred))

# 혼동행렬 시각화
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('confusion_matrix')
plt.xlabel('predict')
plt.ylabel('actual')
plt.show()
