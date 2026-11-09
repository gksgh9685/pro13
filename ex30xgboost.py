# XGBoost : Boosting 알고리즘을 구현한 분류/예측 모델
# Boosting은 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer

# pip install xgboost, lightgbm
import xgboost as xgb
from lightgbm import LGBMClassifier  # xgboost보다 성능은 우수하나 자료 적으면 과적합 발생
import lightgbm as lgb

data = load_breast_cancer()
x=pd.DataFrame(data.data, columns=data.feature_names)
y=data.target
print(x[:3], x.shape)
print(y[:3])

print('레이블 분포 : ', \
    {name:(y==i).sum() for i, name in enumerate(data.target_names)}
)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=12, stratify=y
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 모델 1
xgb_clf = xgb.XGBClassifier(
    booster = 'gbtree', 
    max_depth=6, 
    n_estimate=200,
    eval_metric = 'logloss',
    random_state = 42
)
xgb_clf.fit(x_train,y_train)

# 모델2
lgb_clf = LGBMClassifier(
    n_estimators=200, random_state=42, verbose = -1
)
lgb_clf.fit(x_train, y_train)

# 두 모델에 예측/평가
pred_xgb = xgb_clf.predict(x_test)
pred_lgb = lgb_clf.predict(x_test)
print(f'XGBClassifier acc :  {accuracy_score(y_test, pred_xgb):.5f}')
print(f'LGBClassifier acc :  {accuracy_score(y_test, pred_lgb):.5f}')

print()
# feature 중요도 : gain 기준으로 통일
booster = xgb_clf.get_booster()
xgb_gain = pd.Series(booster.get_score(importance_type='gain'))
lgb_gain = pd.Series(
    lgb_clf.booster_.feature_importance(importance_type='gain'),
    index=x_train.columns   # 숫자 배열 반환 대신 실제 칼럼을 반환
)


# 각 feature의 기여도를 비율로 확인
# 혹시 모든 중요도가 0인 경우 0으로 나누는 오류 방지
xgb_gain_pct = 100 * xgb_gain / (xgb_gain.sum() if xgb_gain.sum() != 0 else 1)
lgb_gain_pct = 100 * lgb_gain / (lgb_gain.sum() if lgb_gain.sum() != 0 else 1)
# print(xgb_gain_pct)
# print(lgb_gain_pct)

# 사용되지 않는 feature은 0으로 채우기
xgb_gain_pct = xgb_gain_pct.reindex(x_train.columns).fillna(0)
lgb_gain_pct = lgb_gain_pct.reindex(x_train.columns).fillna(0)

comp_df = pd.DataFrame({
    'XGBoost(gain %)' : xgb_gain_pct,
    'LightGBM(gain %)' : lgb_gain_pct
}).sort_values('XGBoost(gain %)', ascending=False)
print(comp_df.head(10))  # 중요 변수


# 시각화
topk = 5
top = comp_df.head(topk)[::-1]
fig, axes = plt.subplots(1,2, figsize=(8,5))
xmax = float(np.ceil(top.max().max()))

for ax, col in zip(axes, ['XGBoost(gain %)', 'LightGBM(gain %)']):
    ax.barh(top.index, top[col])
    ax.set_title(f'{col.split()[0]} Feature importance')
    ax.set_xlabel('importance')
    ax.set_xlim(0,xmax)

plt.tight_layout()
plt.show()