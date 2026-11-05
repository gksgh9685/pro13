# decisiontree(의사결정나무) 분류 모델
# 데이터 균일도에 따른 규칙기반의 결정트리
# 트리는 데이터를 직각(수직, 수평) 기준으로 나누면서 영역을 만든다.
# 의사결정나무(decision tree)는 데이터에 있는 규칙이나 패턴을 if-then 형태의 나무 구조로 나타내어 분류나 회귀를 수행하는 지도학습모델


from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, random_state=42)

# n_informative : 분류에 더 적합한 독립변수 2개 생성
print(x[:3])
print(y[:3])

# 지니 불순도(gini impurity)를 기준으로, 최대 깊이 3으로 제한하는 분류 모델을 생성
# gini impurity는 하나의 노드에 서로 다른 클래스가 얼마나 섞여 있는지를 나타냄.
# 예) class 0(100%) class1(0%) : 완전 순수,
#     class 0(50%) class1(50%) : 불순도 가장 높음
model = DecisionTreeClassifier(criterion='gini', max_depth=3)   # 트리 최대 깊이
# 깊이가 너무 크면 학습 데이터에 과적합될 수 있고, 너무 작으면 과소적합될 수 있다.

model.fit(x,y)

# 시각화
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=['x1','x2'], class_names=['0','1'], filled=True)
plt.show()

# 결정 경계 시각화

xx, yy = np.meshgrid(   # x축, y축 값을 조합해서 좌표격자를 생성
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),  # x1 범위를 100개 구간으로 나눔
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
)
# 모든 좌표에 대해 예측값 계산
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape) # 예측 결과를 원래 grod로 변환
print(z)

plt.contour(xx, yy, z, alpha=0.3)   # 영역을 색으로 채워 결정경계를 표현
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('dection boundry')
plt. xlabel('x1')
plt. ylabel('x2')
plt. show()


