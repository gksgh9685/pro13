# 중요도 교차검증도
#[Randomforest 문제1] 
# kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)

# dataset은 winequality-red.csv 

# https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv

 

# winequality-red.csv
# 83.58KB
 

# Input variables (based on physicochemical tests):
#  1 - fixed acidity
#  2 - volatile acidity
#  3 - citric acid
#  4 - residual sugar
#  5 - chlorides
#  6 - free sulfur dioxide
#  7 - total sulfur dioxide
#  8 - density
#  9 - pH
#  10 - sulphates
#  11 - alcohol
#  Output variable (based on sensory data):
#  12 - quality (score between 0 and 10)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('winequality-red.csv')
print(df.head(3))
print(df.info())
print(df.isnull().any()) #결측치 확인 결측치가 없는것을 확인할수 있음

#독립변수
df_x = df.drop(columns='quality')
# 종속변수
df_y = df['quality']

# 학습 테스트분리 7:3
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=12, stratify=df_y)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) #(1117, 11) (479, 11) (1117,) (479,)로 분리 완료

#모델 생성
model = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=12)

# 학습
model.fit(train_x, train_y)
pred = model.predict(test_x)
print('예측값:', pred[:10])
print('실제값:', np.array(test_y[:10]))
print('맞춘 개수:', sum(test_y == pred))
print('분류 정확도:', accuracy_score(test_y, pred))

# 교차 검증 5겹 검증
# 교차검증: 학습 데이터에서 5겹 교차검증
cross_vali = cross_val_score(model, train_x, train_y,cv=5)
print(cross_vali)
print('교차 검증 평균 정확도:', np.round(np.mean(cross_vali), 4))

# 변수 중요도
importances = model.feature_importances_

feature_df = pd.DataFrame({
    'feature': df_x.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_df)
#            feature  importance
# 10               alcohol    0.140626
# 9              sulphates    0.109748
# 6   total sulfur dioxide    0.105147
# 1       volatile acidity    0.103357
# 7                density    0.095232
# 4              chlorides    0.083033
# 8                     pH    0.077360
# 0          fixed acidity    0.075817
# 2            citric acid    0.074514
# 3         residual sugar    0.068394
# 5    free sulfur dioxide    0.066773

#중요도 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
plt.figure(figsize=(8,5))
sns.barplot(x='importance', y='feature', data=feature_df, orient='h')
plt.xlabel('중요도 점수')
plt.ylabel('독립 변수')
plt.show()
plt.close()

