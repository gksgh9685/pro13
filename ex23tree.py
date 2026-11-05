# decisiontree 분류 모델
# 키, 머리카락 길이로 남녀 구분

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np

x = [[180, 15], [177, 42], [156, 35], [174, 65], [161, 25], [160, 45], [170, 65], [155, 55], [168, 5], [188, 10]]
y = ['man', 'woman','woman', 'man','woman', 'woman','woman', 'woman','man', 'man']
print(x)
print(y)
feature_names = ['height', 'hair_length']
class_names = ['man', 'woman']

model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
model.fit(x,y)

# 모델 성능 점수
print('예측값 :', model.predict(x))
print('실제값 : ', y)
print(f'정확도 : {model.score(x,y)}')   # 1.0

# 새로운 데이터로 예측
new_data = [[177, 78]]
print('new_data pred : ', model.predict(new_data))  # new_data pred :  ['woman']

# 시각화
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=feature_names, class_names=model.classes_, filled=True, rounded=True, fontsize=12)
plt.title('decision tree')
plt.show()










