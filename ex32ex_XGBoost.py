"""
xgb_roc_score2: 0.84268
예측값:  [0 0 0 0 0]
실제값:  [0 0 0 0 0]
분류 정확도:  0.9584319915811629
recall(재현율):  0.00473186119873817
RI	Na	Mg	Al	Si	K	Ca	Ba	Fe	
성분들의 종류에 따라 유리의 Type이 변경
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
# pip install xgboost, lightgbm
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from lightgbm import LGBMClassifier # xgboost 보다 성능은 우수하나 자료  적으면 과적합 발생
import lightgbm as lgb

data=pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/glass.csv")
print(data.head(2))
print(data.shape)
# RI     Na    Mg    Al     Si     K    Ca   Ba   Fe  Type
# 0  1.52101  13.64  4.49  1.10  71.78  0.06  8.75  0.0  0.0     1
# 1  1.51761  13.89  3.60  1.36  72.73  0.48  7.83  0.0  0.0     1
# (214, 10)

# feature / label 분리
x_features = data.iloc[:, :-1]   # 모든 행, 마지막 열(TARGET)을 제외한 모든 열
y_lab = data.iloc[:, -1]       # 모든 행, 마지막 열(TARGET) 하나 -> Series

# XGBClassifier는 자동 레이블 변환 안되서 바꿔줘야함
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enco = le.fit_transform(y_lab) 
print(le.classes_)                   

# train,test split
x_train,x_test,y_train,y_test=train_test_split(x_features,y_enco, test_size=0.2, random_state=0)
print("\n\n종속변수 형태",x_train.shape,x_test.shape) #(171, 9) (43, 9)

# 모델
xgb_clf=XGBClassifier(n_estimators=30, random_state=0,  eval_metric='mlogloss') # XGBClassifier 객체
xgb_clf.fit(x_train,y_train, eval_set=[(x_test,y_test)], verbose=True) # 모델 객체 학습 시키기
xgb_roc_score=roc_auc_score(y_test,xgb_clf.predict_proba(x_test), multi_class='ovr')
# multi_class must be in ('ovo', 'ovr')

print(f'xgb_roc_score : {xgb_roc_score:.5f}')

pred=xgb_clf.predict(x_test)
print("예측값: ", pred[:7])
print("실제값: ", y_test[:7])
print('분류 정확도: ', accuracy_score(y_test,pred))


