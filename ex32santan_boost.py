# 산탄데르 은행의 관련 데이터로 고객만족 여부 분류
# label(class, y)명은 target이고 0:만족, 1:불만족

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier  # xgboost보다 성능은 우수하나 자료 적으면 과적합 발생
from xgboost import plot_importance

df=pd.read_csv('train_bank.csv', encoding='latin-1')
print(df.head(2))
print(df.shape)
print(df.info())

# 전체 데이터에서 만족과 불만족의 비율
print(df['TARGET'].value_counts())
unsatisfied_cnt = df[df['TARGET'] ==1].TARGET.count()
total_cnt = df['TARGET'].count()
print(f'불만족의 비율은 {unsatisfied_cnt / total_cnt}')

pd.set_option('display.max_columns', None)
print(df['var3'].describe())
df['var3'].replace(-999999,2,inplace=True)
print(df['var3'].describe())
df.drop('ID', axis=1, inplace=True)
print(df.describe())
print(df.shape)

# feature / label 분리
x_features = df.iloc[:, :-1]
y_label = df.iloc[:, -1]

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape)
train_cnt = y_train.count()
test_cnt = y_test.count()
print('학습데이터 레이블값 분포 비율:', y_train.value_counts() / train_cnt)
print('데이터 레이블값 분포 비율:', y_train.value_counts() / test_cnt)


print()
# 모델
xgb_clf = XGBClassifier(n_estimators =100, random_state = 12, eval_metric ='auc')
xgb_clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)
# 학습은 x_train, y_train로 진행하고, 학습 도중 x_test, y_test로 성능 확인
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(x_test)[:, 1])
print(f'xgb_roc_score : {xgb_roc_score:.5f}')
print()
pred = xgb_clf.predict(x_test)
print('예측값 :', pred[:5])
print('실제값 :', y_test[:5].values)
print('분류 정확도 : ', accuracy_score(y_test, pred))
# roc_auc : 만족과 불만족을 얼마나 잘 구분하는가?
# accuracy : 전체 고객중 얼마나 맞혔는가?
# recall : 실제 불만족 고객을 얼마나 찾아냈는가?
# precision : 불만족이라고 예측한 고객 중 실제 불만족 비율은?

print('\nGridSearchCV로 최적의 파라미터 찾기 ---')
from sklearn.model_selection import GridSearchCV

xgb_clf = XGBClassifier(n_estimators = 10)
# max_depth: 트리 깊이, min_child_weight:관측치 가중치합 최소, colsample_bytree:피처 비율
params = {'max_depth':[5,7],
        'min_child_weight':[1,3],
        'colsample_bytree':[0.5, 0.75]
}
gridcv= GridSearchCV(xgb_clf, param_grid=params)
gridcv.fit(x_train, y_train, eval_set = [(x_test, y_test)])
print('gridcv 최적 파라미터 : ', gridcv.best_params_)

print()
best_clf = XGBClassifier(n_estimators=10, random_state=12,
        colsample_bytree=0.75, max_depth=5, min_child_weight=1)
best_clf.fit(x_train, y_train)

xgb_roc_score2 = roc_auc_score(y_test,
                            best_clf.predict_proba(x_test)[:,1])
print(f'xgb_roc_score2 : , {xgb_roc_score2:.5f}')

pred2= best_clf.predict(x_test)
print('예측값 : ',pred2[:5])
print('실제값 : ', y_test[:5].values)
print('분류 정확도 : ', accuracy_score(y_test, pred2))

from sklearn.metrics import recall_score
recall = recall_score(y_test, pred2)
print('recall(재현율) : ', recall)

