# iris dataset으로 실습
# 다층 인공신경망

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 표준화, 정규화
from sklearn.linear_model import LogisticRegression
# LogisticRegression : 이항분류 뿐 아니라 다중 클래스를 지원함

iris = datasets.load_iris()
print(iris.keys())
print(iris.target)
print(iris.data[:3])
print(np.corrcoef(iris.data[:,2], iris.data[:,3])[0, 1])  # 0.96286543

x = iris.data[:, [2, 3]]   # 꽃잎 length, width
y = iris.target
print(x.shape, y.shape)  # (150, 2) (150,)
print(x[:3], y[:3], set(map(int, y)))   # {0, 1, 2}

print('\ntrain / test split (7:3)')
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)  # (105, 2) (45, 2) (105,) (45,)
print(x_train[:3],x_test[:3],y_train[:3],y_test[:3])

"""
# Scaling (데이터 크기 표준화/정규화 - 최적화 과정에서 안정성, 수렴속도 향상, 과적합/과소적합 방지 등의 효과)
sc = StandardScaler()    # 표준화
sc.fit(x_train)
sc.fit(x_test)    # 주의 : feature만 스케일링 한다.  label X
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])
print(x_test[:3])
# 스케일링 원복
ori_x_train = sc.inverse_transform(x_train)
print("원복값 : ", ori_x_train[:3])
"""

print('분류 모델 생성 ----')
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(
    hidden_layer_sizes=(20, 10),
    activation='relu',
    solver='adam',  # 손실 최소화 함수
    learning_rate_init=0.001,  # 학습률
    max_iter=1000,
    random_state=42,
    verbose=1
)
print(model)

model.fit(x_train, y_train)  # train으로 학습

# 분류 예측
y_pred = model.predict(x_test)   # x_test로 예측
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print(f'총갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')
# 총갯수:45, 오류수:1

print(f'분류 정확도 : {accuracy_score(y_test, y_pred)}') # 0.97777
conf_mat = confusion_matrix(y_test, y_pred)
print('혼동행렬 : \n', conf_mat)
print('분류 정확도 : ', np.trace(conf_mat) / np.sum(conf_mat))  # 클래스가 다항인 경우
# np.trace(conf_mat)은 혼동행렬 대각선의 합

print(f'train score : {model.score(x_train, y_train)}') # 0.9714285
print(f'test score : {model.score(x_test, y_test)}')    # 0.9777777
# train score와 test score의 차이가 크면 과적합 의심.

# 학습 후 검증이 끝난 모델 저장 후 읽기
import joblib
joblib.dump(model, 'ex20model.sav')   # .model

# ---------------------------------------------
read_model = joblib.load('ex20model.sav')
print(read_model)  # LogisticRegression(random_state=0)

print('새 값으로 예측하기')
new_data = np.array([[5.5, 2.2],[0.8, 0.9], [8.5, 6.5]])
# 주의 : 만약 표준화된 feature로 모델을 생성했다면 new_data도 표준화 해야 함.
# sc = StandardScaler()
# sc.fit(new_data)
# new_data = sc.transform(new_data)

new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred)     # [2] : 확률값 중 가장 큰 인덱스 번호
print(read_model.predict(new_data))  # [[2.53496435e-05 5.43511967e-02 9.45623454e-01]]

# iris dataset 분류 연습용 시각화 코드
import matplotlib.pyplot as plt

import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')      # 마커 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)   # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)   
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regionFunc(X=x_combined_std, y=y_combined, classifier=read_model, test_idx=range(105, 150), title='scikit-learn제공') 