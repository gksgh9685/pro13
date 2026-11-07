# 앙상블 학습:여러 개의 분류기를 생성하고, 그 예측을 결합함으로써 보다 정확한 예측결과를 얻음
# 강력한 하나의 모델 보다는 약한 모델 여러 개를 조합하여 더 정확한 예측결과를 도출

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from collections import Counter

cancer = load_breast_cancer()
print(cancer.keys())
x, y = cancer.data, cancer.target
print(x[:2])
print(y[:2])
print(np.unique(y)) # [0 1]
print(cancer.target_names)
print('악성 종양 수 :', np.sum(y==0))   # 악성 종양 수 : 212
print('양성 종양 수 :', np.sum(y==1)) # 양성 종양 수 : 357

# 0과 1의 비율
counter = Counter(y)
total = sum(counter.values())
for cls, cnt in counter.items():
    print(f'class {cls}:{cnt}개 ({cnt/total:.2f})')
# class 0:212개 (0.37)
# class 1:357개 (0.63)

print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12, stratify=y)

# stratify=y : train, testy비율 유지(불균형한 데이터 모델 평가 시 왜곡방지)

y_li = y.tolist()
ytr_li = y_train.tolist()
yte_li = y_test.tolist()
print('전체 분포 : ', Counter(y_li))    # 전체 분포 :  Counter({1: 357, 0: 212})
print('train 분포 : ', Counter(ytr_li)) # train 분포 :  Counter({1: 285, 0: 170})
print('test 분포 : ', Counter(yte_li))  # test 분포 :  Counter({1: 72, 0: 42})

print()
# 개별 모델 작성
# make_pipeline : 전처리와 모델을 일체형 객체로 관리
logi = make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000, solver='lbfgs', random_state=12))
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

# 트리 계열 모델은 스케일이 달라도 분할 로직에는 영향을 주지 않는다. (표준화, 정규화 안해도 됨)
tree = DecisionTreeClassifier(max_depth=5, random_state=12)

# 앙상블 모델
votingModel = VotingClassifier(
    estimators=[('LR', logi), ('KNN', knn),('DT',tree)], voting='soft'
)

# 개별 모델 성능 확인
named_models = [('LR', logi), ('KNN',knn),('DT',tree)]
for name, clf in named_models:
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(f'{name} 정확도 : {accuracy_score(y_test, pred):.4f}')
# LR 정확도 : 0.9912
# KNN 정확도 : 0.9737
# DT 정확도 : 0.8772

# votingmodel 성능
votingModel.fit(x_train, y_train)
vpred = votingModel.predict(x_test)
print(f'votingModel 분류기 정확도 : {accuracy_score(y_test, vpred):.4f}')   # votingModel 분류기 정확도 : 0.9649

print()
# 선택적 작업 : breast_cancer 데이터가 다소 불균형을 이루므로 stratifiedkfold 사용 가능
cvfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
cv_score = cross_val_score(votingModel, x, y, cv=cvfold, scoring='accuracy')
print(f'보팅 5겹 cv 정확도 평균 : {cv_score.mean():.4f} (표준편차: +- {cv_score.std():.4f})')
# 

print()
# 모델 성능 지표
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score
print('\n보팅 모델 상세 평가')
print(classification_report(y_test, vpred))
print('confusion_matrix :\n', confusion_matrix(y_test, vpred))
print('roc_auc_score : \n', roc_auc_score(y_test, votingModel.predict_proba(x_test)[:,1]))




