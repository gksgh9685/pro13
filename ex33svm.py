# SVM 알고리즘은 머신러닝 알고리즘에서 가장 유명하고 많이 사용되고 있다.
# SVM 알고리즘은 기본적으로 두 개의 그룹(데이터)을 분리하는 방법으로 
# 데이터들과 거리가 가장 먼 초평면(hyperplane)을 선택하여 분리하는 방법.

# : 결정 경계에 가장 가까이 위치하여 마진을 결정하고 초평면을 지탱하는 
# 핵심 데이터 포인트들이다. 
# 전체 데이터 중 서포트 벡터를 제외한 나머지는 모델 결정에 영향을 주지 않는다.

# Support vector 확인해 보기
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np

x, y = make_blobs(n_samples=50, centers=2, cluster_std=0.5, random_state=4)
print(x[:5])
y = 2 * y - 1   # 0, 1 -> -1, 1 
print(y[:5])

plt.scatter(x[y == -1, 0], x[y == -1, 1], marker='o', label='-1 클래스')
plt.scatter(x[y == +1, 0], x[y == +1, 1], marker='x', label='+1 클래스')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

# SVM 모델
from sklearn.svm import SVC

model = SVC(kernel='linear', C=1.0).fit(x, y)
xmin = x[:, 0].min()
xmax = x[:, 0].max()
ymin = x[:, 1].min()
ymax = x[:, 1].max()
xx = np.linspace(xmin, xmax, 10)
yy = np.linspace(ymin, ymax, 10)
x1, x2 = np.meshgrid(xx, yy)  # 두 개의 1차원 배열로 2차원 좌표 격자(grid) 생성

# a = np.array([1,2,3])
# b = np.array([10,20])
# a, b = np.meshgrid(a, b)
# print(a)   # [[1 2 3][1 2 3]]
# print(b)   # [[10 10 10][20 20 20]]

z = np.empty(x1.shape)

for (i, j), val in np.ndenumerate(x1):  # 배열좌표와 값 쌍을 생성하는 반복기 반환
    x_value = val
    y_value = x2[i, j]
    p = model.decision_function([[x_value, y_value]])
    z[i, j] = p[0]

plt.scatter(x[y == -1, 0], x[y == -1, 1], marker='o', label='-1 클래스')
plt.scatter(x[y == +1, 0], x[y == +1, 1], marker='x', label='+1 클래스')
plt.contour(x1, x2, z, levels=[-1, 0, 1], colors='k', 
            linestyles=['dashed', 'solid','dashed'])   # 등고선 그리기 - 결정경계 시각화 목적
plt.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1], 
            s=300, alpha=0.5)
x_new = [10, 2]
plt.scatter(x_new[0], x_new[1], marker='^', s=100)
plt.text(x_new[0] + 0.03, x_new[1] + 0.08, '테스트데이터')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('SVM 예측 결과')
plt.show()

print('support vector : \n', model.support_vectors_)
#  [[9.03715314 1.71813465][9.17124955 3.52485535]]
