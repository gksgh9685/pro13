# logisticregression : 날짜 예보(비가 올지 안올지)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv')
print(data.head(2), data.shape) # (366, 12)
print(data.info())
df = pd.DataFrame()
df = data.drop(['Date', 'RainToday'], axis=1)
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, 'No':0})
print(df.head(2), df.shape)     # 366, 10
print(df['RainTomorrow'].unique())  # [1 0]
# raintomorrow : 종속변수(class) - 범주형, 나머지열 : 독립변수(feature)

print('데이터 분리 : 학습용(train dataset), 검증용(test dataset)')
# 모델의 성능을 객관적으로 파악하기 위함. - 과적합 방지
train, test = train_test_split(df, test_size=0.3, random_state=42)
print(train.shape, test.shape)  # (256, 10) (110, 10)
print(train.head(3))
print(test.head(3))

# 모델 생성
col_select = '+'.join(train.columns.difference(['RainTomorrow']))
print(col_select)
my_formula = 'RainTomorrow ~ ' + col_select
model = smf.logit(formula=my_formula, data=train).fit()
print(model.summary())
print(model.params)
print()
print('예측값 : ', np.rint(model.predict(test)[:5].values))
print('실제값 : ', test['RainTomorrow'][:5].values)

# 분류 정확도(train data로 진행)
# pred_table() : 기본적으로 모델 학습에 사용한 train 데이터만 대상으로 혼동행렬을 생성
pred_mat = model.pred_table()
print(pred_mat)
print('분류 정확도(Accuracy) : ',(pred_mat[0][0] + pred_mat[1][1]) / len(train))    #  0.87109375

# 분류 정확도(test data로 진행)
from sklearn.metrics import accuracy_score, confusion_matrix
pred = model.predict(test)
conf_mat = confusion_matrix(test['RainTomorrow'], np.rint(pred))  # 실제값, 예측값
print(conf_mat)
print('분류 정확도(Accuracy) : ', (conf_mat[0][0] + conf_mat[1][1]) / len(test))
print('분류 정확도(Accuracy) : ', accuracy_score(test['RainTomorrow'], np.rint(pred)))
print('분류 정확도(Accuracy) : ', accuracy_score(test['RainTomorrow'], (pred >= 0.5).astype(int)))

# 트레인으로 학습한 정확도와 테스트로 진행한 정확도가 비슷하면 잘된 모델
# 트레인의 정확도와 테스트의 정확도가 차이가 많이 난다 이때 트레인의 적합도가 더 높으면 과적합으로 문제임.

