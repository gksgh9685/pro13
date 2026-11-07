# RandomForest 분류 알고리즘
# 머신러닝에서 분류(Classification)와 회귀(Regression) 분석에 널리 사용되는
# 강력한 앙상블(Ensemble) 학습 알고리즘입니다.
# 여러 개의 결정 트리(Decision Tree)를 생성하고, 이들의 예측 결과를 종합하여
# 최종적인 분류 결과를 도출합니다.
# 앙상블 기법 중 배깅(Bagging, Bootstrap Aggregation)을 사용
# : 복수의 샘플 데이터와 DecisionTree를 학습시키고 결과를 집계하는 방식
# 참고 : 우수한 성능은 Boosting, 과적합이 걱정된다면 Bagging

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
print(df.head(2))
print(df.info())
print(df.shape)
print(df.isnull().any())

df = df.dropna(subset=['Pclass','Age','Sex'])
print(df.shape)

df_x = df[['Pclass','Age','Sex']].copy()
print(df_x.head(3))
# 전처리 -- sex열(male, female) -> label encoding(문자범주형 -> 점수)
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
# df_x.loc['Sex'] = df_x['Sex'].astype('object')   # 정수 대입 가능 자료형으로 변경
df_x['Sex'] = encoder.fit_transform(df_x['Sex'])
print(df_x.head(3))

df_y = df['Survived']   # label(class)
print(df_y.head(3))

print()
train_x,test_x,train_y,test_y = train_test_split(df_x, df_y, random_state=12)
print(train_x.shape, test_x.shape,train_y.shape, )

# 모델 생성
model = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=12)
# criterion='entropy'도 가능, decisiontrr의 갯수 --> n_estimators=100이 기본
print(model) #RandomForestClassifier(n_estimators=500, random_state=12)
model.fit(train_x, train_y)

pred = model.predict(test_x)
print('예측값:',pred[:5])
print('실제값:',np.array(test_y[:5]))
print('맞춘 갯수:',sum(test_y == pred))
print('전체 대비 맞춘 비율 :',sum(test_y == pred) / len(test_y))
print('분류 정확도:', accuracy_score(test_y,pred))

# 교차 검증 (kfold)
cross_vali = cross_val_score(model, df_x, df_y, cv=5)
print(cross_vali)
print('교차 검증 평균 정확도 :', np.round(np.mean(cross_vali), 4))

# randomforestclassifier는 중요 변수 제공
print('중요 변수 확인하기 ---')
print('특성(변수, feature) 중요도 :', model.feature_importances_)
# feature_importances_ : 각 특성이 예측해 기여한 정도(중요도)를 수치로 변환
# 중요도 값의 합은 1.0, 수치가 클수록 해당 변수가 불순도 감소에 더 많이 기여함
# print(0.16172779 + 0.49842824 + 0.33984396) # 약 1.0

# 시각화
import matplotlib.pyplot as plt
import koreanize_matplotlib
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('변수별 중요도 점수')
plt.ylabel('독립 변수')
plt.yticks(np.arange(n_features), df_x.columns)
plt.show()
plt.close()

print('전체 변수 대상으로 중요도 확인')
# ticket, cabin, emvarked : 문자형 - 바로 사용 불가(encoding 필요)
# passengerid, name : survived와 상광 없음

df_imsi = df[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch']].copy()
# df_imsi.loc['Sex'] = df_imsi['Sex'].astype('object')
df_imsi['Sex'] = encoder.fit_transform(df_imsi['Sex'])
train_x,test_x,train_y,test_y = train_test_split(df_imsi, df_y, test_size=0.3, random_state=12)
model.fit(train_x, train_y)

importances = model.feature_importances_

# 컬럼명 + 중요도
feature_df = pd.DataFrame({
    'feature':df_imsi.columns,
    'importances':importances
}).sort_values(by='importances', ascending=False)

print(feature_df)

# 시각화
import seaborn as sns

plt.figure(figsize=(8,5))
sns.barplot(x='importances', y='feature', data=feature_df, orient='h')
plt.xlabel('변수별 중요도 점수')
plt.ylabel('독립 변수')
plt.yticks(np.arange(n_features), df_x.columns)
plt.show()
plt.close()