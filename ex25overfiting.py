# 과적합 방지 방법 이해
# train / test split : 일반화 성능 향상
# k-Fold : 안정적 평가
# gridsearchcv : 최적의 하이퍼파라미터 검색

# iris 사용

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
print(iris.keys())

train_data = iris.data
train_label = iris.target
print(train_data[:3])
print(train_label[:3])

# 분류 모델
df_clf = DecisionTreeClassifier()
df_clf.fit(train_data, train_label)
pred = df_clf.predict(train_data)   # 학습 데이터로 검증
print('예측값 : ', pred)
print('실제값 : ', train_label)
print('분류 정확도 :', accuracy_score(train_label, pred))   # 1.0 과적합 의심

print('\n과적합 방지 목적 처리 1: train / test split')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=12
)
df_clf.fit(x_train, y_train)    # train data로 학습
pred2 = df_clf.predict(x_test)   # train data로 학습
print('예측값(split) :', pred2)
print('실제값(split) :', y_test)
print('분류 정확도(split) :', accuracy_score(y_test, pred2))    # 분류 정확도(split) : 0.9555555555555556

print('\n과적합 방지 목적 처리 2: 교차 검증(k-fold cross validation)')
# k-폴드 교차 검증은 머신러닝 모델의 일반화 성능을 객관적으로 평가하기 위해
# train 데이터를 k개의 부분집합(fold)으로 나누어 학습과 검증을 반복하는 방법.
# train dataset = train + validation dataset으로 나누어 모델을 학습
from sklearn.model_selection import KFold
import numpy as np

features = iris.data
label = iris.target
df_clf2 = DecisionTreeClassifier(criterion='gini',max_depth=3, random_state=12)

Kfold = KFold(n_splits=5) # k:5회 접기
cv_acc = [] # 각 1회 마다 정확도 저장용
print('iris shape : ', features.shape)  # iris shape :  (150, 4)
# kFold 학습시 전체 150행이 학습데이터(4/5, 120개), 검증데이터(1/5, 30개)로 분할되어 학습함.

n_iter = 0

for train_index, test_index in Kfold.split(features):
    # print('n_iter(반복수): ', n_iter)
    # print('train_index:', train_index)
    # print('test_index:', test_index)
    # n_iter += 1
    xtrain, xtest = features[train_index], features[test_index]
    ytrain, ytest = label[train_index], label[test_index]
    # 학습 및 예측
    df_clf2.fit(xtrain, ytrain)     # train으로 학습
    pred = df_clf2.predict(xtest)   # 모델 생성을 위한 학습 도중 예측 - 검증 목적
    n_iter += 1
    # 반복할 때 마다 정확도 출력
    acc = np.round(accuracy_score(ytest, pred), 5)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print(f'반복수 : {n_iter}, 교차검증 정확도:{acc}, 학습데이터크기 :{train_size}, 검증크기:{test_size}')
    print(f'반복수:{n_iter}, 검증데이터 인덱스:{test_index}')

    cv_acc.append(acc)

print('cs_acc : ', np.array(cv_acc).astype(float))
print('평균 검증 정확도 :', np.mean(cv_acc))

# 참고 : stratifiedKFold - 불균형 분포도를 가진 레이블 데이터 집합을 처리하기 위한kfold 방식
# 예를 들어 대출 사기 데이터인 경우 대부분은 정상, 사기 레이블은 극히 일부임

print('\n과적합 방지 목적 처리 2-1 : 교차 검증 단순화 함수 사용')
# cross_val_score를 양용해 교차 검증을 간단히 수행
from sklearn.model_selection import cross_val_score
data = iris.data
label = iris.target

score = cross_val_score(df_clf2, data, label, scoring='accuracy', cv=5)
print('교차 검증별 정확도 :', np.round(score, 3))
print('평균 검증 정확도 :', np.round(np.mean(score), 3))

print('\n과적합 방지 목적 처리 3 : gridsearchcv')
# 과적합 방지 간접 방법
# 최적의 파라미터 찾기(내부적으로 kfold를 사용해 과적합을 줄이는데 도움을 준다)
from sklearn.model_selection import GridSearchCV
# DecisionTreeClassifier(*
#     criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
#     splitter: Literal['best', 'random'] = "best",
#     max_depth: Int | None = None,
#     min_samples_split: float = 2,
#     min_samples_leaf: float = 1,
#     min_weight_fraction_leaf: Float = 0,
#     max_features: float ... )
# 편의상 일부 파라미터만 사용
# max_depth : 노드의 최대 길이
# min_samples_split : 노드 분할을 위한 최소한의 샘플수 (과적합 제어와 관련)

parameters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}
grid_dtree = GridSearchCV(estimator=df_clf2, param_grid=parameters, cv=3, refit=True)
grid_dtree.fit(x_train, y_train)
# 내부적으로 복수 개의 모형을 생성하고, 이를 실행시켜 최적의 파라미터를 찾아줌

import pandas as pd
pd.set_option('display.max_columns', None)
score_df = pd.DataFrame(grid_dtree.cv_results_) # 결과를 dataframe에 저장
print(score_df)

print('gridsearchcv 최적 파라미터:', grid_dtree.best_params_)
print('gridsearchcv 최적 정확도:', grid_dtree.best_score_)
# gridsearchcv 최적 파라미터: {'max_depth': 3, 'min_samples_split': 2}
# gridsearchcv 최적 정확도: 0.9238095238095237

# 추천된 파라미터로 최적의 모델을 생성(직접 생성 또는 best_estimator_ 사용)
bestModel = grid_dtree.best_estimator_
print(bestModel)
best_pred = bestModel.predict(x_test)
print('예측 결과 : ', best_pred)
print('정확도 : ', accuracy_score(y_test, best_pred))   # 정확도 :  0.9555555555555556