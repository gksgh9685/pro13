# LogisticRegression : 정성적 분류 모델
# 선형결합을 로그오즈(logit())로 해석하고, 이를 시그모이드 함수를 통해 확률로 변환
# 이항분류(다항분류도 가능) - 독립변수(feature) : 연속형, 종속변수(label, class)
# 참고 : logisticregression을 근거로 뉴럴넷의 뉴런이 사용~

# mtcarts dataset을 사용
import statsmodels.api as sm

mtcarsdata = sm.datasets.get_rdataset('mtcars')
print(mtcarsdata.keys())
mtcars = mtcarsdata.data
print(mtcars[:2])
print(mtcars.info())

print()
# 연비와 마력수에 따른 변속기 분류(수동, 자동)
mtcar = mtcars.loc[:,['mpg','hp','am']]
print(mtcar.head(3))    
print(mtcar['am'].unique()) # [1(수동) 0(자동)]

# 모델 작성 방법 1 : logit()
import numpy as np
import statsmodels.formula.api as smf

formula = 'am ~ hp + mpg' # 범주형 ~ 연속형 + ...
result = smf.logit(formula=formula, data=mtcar).fit()
print(result.summary()) # Logit Regression Results

pred = result.predict(mtcar[:10])
print('예측값 :', pred.values)  # 예측값 : [0.25004729 0.25004729 0.55803435 0.35559974 0.39709691 0.00651918  0.10844152 0.63232168 0.58498645 0.06598365]
print('예측값 :', np.around(pred.values))   # 예측값 : [0. 0. 1. 0. 0. 0. 0. 1. 1. 0.]
print('실제값 :', mtcar['am'][:10].values)  # 실제값 : [1 1 1 0 0 0 0 0 0 0]

# 혼동 행렬(confusion matrix, 오차 행렬)은 인공지능이나 머신러닝의 분류 모델이
# 예측한 결과가 실제 정답과 얼마나 맞고 틀렸는지를 4가지 경우의 수로 정리한 2x2표
# 현재 모델의 분류 정확도 확인 방법 1=> confusion matrix 이용


# 현재 모델의 분류 정확도 확인방법 1 => confusion matrix 이용
print()
print('수치에 대한 집계표(혼동 행렬)')
conf_tab = result.pred_table()
print(conf_tab)
# [[16.  3.]
#  [ 3. 10.]]
print('분류 정확도 :', (16 +10) / len(mtcar))
print('분류 정롹도 :', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))

# 현재 모델의 분류 정확도 확인방법 2 => accuracy_score() 이용
pred2 = result.predict(mtcar)

from sklearn.metrics import confusion_matrix, accuracy_score

# 예측 확률을 클래스(0 또는 1)로 변환
pred_class = (pred2 > 0.5).astype(int)
conf_mat = confusion_matrix(mtcar['am'], pred_class)
print('confusion_matrix 혼동행렬 :\n', conf_mat)
print('분류 정확도 : ', (conf_mat[0][0] + conf_mat[1][1]) / len(mtcar))

print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(pred2)))

print('** * 10')
# 모델 작성 방법 2 : gim() <-- 일반화된 선형모델 ~ pred_taable() 지원 안함
result2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit()
# binomial() : 이항분포, gaician : 정규분포 - 기본값
print(result2.summary())

glm_pred = result2.predict(mtcar[:10])
print('glm 예측값 : ', np.around(glm_pred.values))
print('glm 실제값 : ', mtcar['am'][:10].values)

glm_pred2 = result2.predict(mtcar)
print('glm 분류 정확도 : ', accuracy_score(mtcar['am'], np.around(glm_pred2)))
# logit() : 로지스틱 회귀 전용 모델, 이진분류, 연결함수는 logit link
# glm() : 일반화 선형모델, 이진분류/개수 데이터 등, 연결함수는 분포에 따라 지정

print('새로운 값으로 예측하기 ---')
import pandas as pd
newdf = pd.DataFrame()
newdf['mpg'] = [10, 30, 120, 130]
newdf['hp'] = [100, 110, 80, 130]
print(newdf)
new_pred = result2.predict(newdf)
print('예측결과 :', np.around(new_pred.values)) # 소수점 지정 가능
print('예측결과 :', np.rint(new_pred.values))   # # 소수점 지정 불가

# 머신러닝의 포용성(inclusion, tolerance)
# 머신러닝 모델을 평가할 때는 훈련 데이터의 100% 정확도라는 '수자'에 집착하기보다,
# '이 모델이 현실의 불확실성과 다양한 변수를 얼마나 유연하게 포용할 수 있는가?'를
# 고려해야함. 수학에서는 100%의 답을 원하지만 ml은 정확도가 95%가 될지라도 주변 값을 포용하도록 하는 것이 중요하다.
# 예를 들어 고양이 분류 모델인 경우 꼬리가 없는 고양이도 분류할 수 있어야 한다.


# 모델의 일반화
# 편향-분산 트레이드오프( bias-variance trade-off)
# 상반관계: 편향과 분산은 일반적으로 반비례 관계에 있다.
# 모델을 단순하게 만들면 편향이 커지고 분산이 작아진다.(과소적합 위험)
# 모델을 복잡하게 만들면 편향이 작아지고 분산이 커진다(과적합 위험)
# 편향과 분산의 합(총 오차)이 최소가 되는 지점,
# 즉 일반화 성능이 가장 좋은 최적의 모델 복잡도를 찾는 것.

# 좋은 모델은 포용성이 있는 일반화된 모델이다.









