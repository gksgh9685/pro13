# 최근접 이웃(k-Nearest Neighbors)
# 알고리즘은 우리가 예측하려고 하는 임의의 데이터와 가장 가까운 거리의
# 데이터 k개를 찾아 다수결에 의해 데이터를 예측하는 방법이다.
# 거리가 가까운 k개의 이웃을 선택할 때 유클리드 거리를 사용한다.

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

train = [
    [5,3,2],
    [1,3,5],
    [4,5,6],
]
label = [0, 1, 1]

train = np.array(train)
plt.scatter(train[:,0], train[:,1], c=label, s=100, cmap='coolwarm')
plt.xlabel('특성 1')
plt.ylabel('특성 2')
plt.show()

# 가까운 이웃 3개를 참조, 가까운 이웃일수록 더 높은 가중치를 부여
kmodel = KNeighborsClassifier(n_neighbors=3, weights='distance')
kmodel.fit(train, label)
pred = kmodel.predict(train)
print('pred :',pred)
print(f'test acc : {kmodel.score(train, label)}')

# 새로운 데이터로 예측
new_data = [[1,2,9],[6,2,1]]
new_pred = kmodel.predict(new_data)
print('new_pred :', new_pred)



