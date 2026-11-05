# 참여 칼럼 : 
#    - Daily Time Spent on Site : 사이트 이용 시간 (분)
#    - Age : 나이,
#    - Area Income : 지역 소득,
#    - Daily Internet Usage :일별 인터넷 사용량(분),
#    - Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# 데이터 간 단위가 큰 경우 표준화 작업을 시도한다.
# 모델 성능 출력 : 정확도, 정밀도, 재현율, ROC 커브와 AUC 출력
# 새로운 데이터로 분류 작업을 진행해 본다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from sklearn.linear_model import LogisticRegression

# 데이터 불러오기
dataset = pd.read_csv('https://raw.githubusercontent.com/Mindy12-ux/python_fork_test/refs/heads/master/testdata_utf8/advertisement.csv')
print('데이터 정보')
print(dataset.head(2))
print(dataset.info())

# x,y 나누기
x = dataset[['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']]
y = dataset['Clicked on Ad']
print(x[:3])

# train/test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
print('\ntrain/test 분리')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 스케일링
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print('\n스케일링 결과')
print(x_train[:3])
print(x_test[:3])

# 분류 모델 생성
model = LogisticRegression(
    C=1.0,
    solver='lbfgs',
    max_iter=100,
    random_state=0
)

model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred[:10])
print('실제값  :', y_test.values[:10])
print(f'총개수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')

print()
# 모델 성능 평가
acc_score = metrics.accuracy_score(y_test, y_pred)
precision_score = metrics.precision_score(y_test, y_pred)
recall_score = metrics.recall_score(y_test, y_pred)
print('acc(정확도) : ', acc_score)
print('precision(정밀도) : ', precision_score)
print('recall(민감도) : ', recall_score)

cl_rep = metrics.classification_report(y_test, y_pred)
print('\n분류모델 성능평가 보고서 : \n', cl_rep)

# 혼동행렬
print()
print('혼동행렬\n', metrics.confusion_matrix(y_test, y_pred))

# ROC 커브
fpr, tpr, thresholds, = metrics.roc_curve(y_test, model.decision_function(x_test))
plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0,1], [0,1], 'k--', label='random classifier line(AUC=0.5)')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve')
plt.legend()
plt.show()

# AUC 계산
print('AUC : ', metrics.auc(fpr, tpr))


# 새로운 데이터로 분류
new_data = np.array([
    [70.23, 27, 5430.39, 234.13],
    [89.23, 40, 6782.34, 192.34],
    [50.34, 58, 8792.34, 100.12],
    [20.11, 60, 9873.45, 10.23]
])

new_data = sc.transform(new_data)
new_pred = model.predict(new_data)

print('예측 결과 : ', new_pred)
print(model.predict_proba(new_data))