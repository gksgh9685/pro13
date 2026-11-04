# [로지스틱 분류분석 문제1]

# 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.

# --------------------------------------
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

# 독립변수 : 소득수준
# 종속변수 : 외식유무 -> 1 or 0

# 데이터 준비: 각 행을 [요일, 외식유무, 소득수준]으로 정리
data = pd.DataFrame([
    ['토', 0, 57],
    ['토', 0, 39],
    ['토', 0, 28],
    ['화', 1, 60],
    ['토', 0, 31],
    ['월', 1, 42],
    ['토', 1, 54],
    ['토', 1, 65],
    ['토', 0, 45],
    ['토', 0, 37],
    ['토', 1, 98],
    ['토', 1, 60],
    ['토', 0, 41],
    ['토', 1, 52],
    ['일', 1, 75],
    ['월', 1, 45],
    ['화', 0, 46],
    ['수', 0, 39],
    ['목', 1, 70],
    ['금', 1, 44],
    ['토', 1, 74],
    ['토', 1, 65],
    ['토', 0, 46],
    ['토', 0, 39],
    ['일', 1, 60],
    ['토', 1, 44],
    ['일', 0, 30],
    ['토', 0, 34],
], columns=['요일', '외식유무', '소득수준'])
# 토요일 또는 일요일인 행만 선택
data = data[(data['요일'] == '토') | (data['요일'] == '일')].copy()
# 주말만 선택(평일은 제외)


# **모델 작성 방법1** : logit()이용
formula = '외식유무 ~ 소득수준'
# 범주형(종속변수) ~ 연속형(독립변수) 설정

result = smf.logit(formula=formula, data = data).fit()

print(result.summary()) # ols표와 같은 형태가 나온다.
# 결과 : 
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:                외식유무   No. Observations:                   21
# Model:                          Logit   Df Residuals:                       19
# Method:                           MLE   Df Model:                            1
# Date:                Wed, 04 Nov 2026   Pseudo R-squ.:                  0.6240
# Time:                        12:40:42   Log-Likelihood:                -5.4648
# converged:                       True   LL-Null:                       -14.532
# Covariance Type:            nonrobust   LLR p-value:                 2.058e-05
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    -11.8668      4.982     -2.382      0.017     -21.632      -2.102
# 소득수준       0.2375      0.100      2.366      0.018       0.041       0.434
# ==============================================================================

logit_pred = result.predict(data[:10])

print('logit 예측값 : ', np.around(logit_pred.values))
print('logit 실제값 : ', data['외식유무'][:10].values)
# 결과 : 
# logit 예측값 :  [1. 0. 0. 0. 1. 1. 0. 0. 1. 1.]
# logit 실제값 :  [0  0  0  0  1  1  0  0  1  1]



# **모델 작성 방법2** : glm()이용
result2 = smf.glm(formula = formula, data=data, family=sm.families.Binomial()).fit()

print(result2.summary())
# 결과 : 
#                  Generalized Linear Model Regression Results                  
# ==============================================================================
# Dep. Variable:                외식유무   No. Observations:                   21
# Model:                            GLM   Df Residuals:                       19
# Model Family:                Binomial   Df Model:                            1
# Link Function:                  Logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -5.4648
# Date:                Wed, 04 Nov 2026   Deviance:                       10.930
# Time:                        12:40:42   Pearson chi2:                     11.7
# No. Iterations:                     7   Pseudo R-squ. (CS):             0.5783
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    -11.8668      4.982     -2.382      0.017     -21.632      -2.102
# 소득수준        0.2375      0.100      2.366      0.018       0.041       0.434
# ==============================================================================
glm_pred = result2.predict(data[:10])

print('glm 예측값 : ', np.around(glm_pred.values))
print('glm 실제값 : ', data['외식유무'][:10].values)
# 결과 : 
# glm 예측값 :  [1. 0. 0. 0. 1. 1. 0. 0. 1. 1.]
# glm 실제값 :  [0  0  0  0  1  1  0  0  1  1]


# 평가 - 모델의 학습과 통계적 적합도 평가 : 가능성(우도)
#      - 분류 정확도 확인 - confusion-matrix
#                        - accuracy_score()를 사용한다

# [분류 정확도 확인방법1 : confusion-matrix(수치에 대한 집계표(혼동 행렬)) - logit()사용]
conf_tab = result.pred_table()
print(conf_tab)
# 결과 : 
#      [[10.  1.]
#       [ 1.  9.]]

print('분류 정확도(logit) : ', (conf_tab[0][0] + conf_tab[1][1]) / len(data))
# 분류 정확도(logit) :  0.9047619047619048

# [분류 정확도 확인방법2 : accuracy_score()이용) - glm()사용]
from sklearn.metrics import accuracy_score
pred2 = result2.predict(data)
print('분류 정확도(glm) : ', accuracy_score(data['외식유무'], np.around(pred2)))
# 분류 정확도(glm) :  0.9047619047619048



# 새로운 값으로 예측하기 ----------------
newdf = pd.DataFrame()
newdf['소득수준'] = [23, 46, 92, 124, 0]
print(newdf)

# glm()모델에 새로운 newdf를 넣기
new_pred = result2.predict(newdf) 

print('예측결과 : ', np.rint(new_pred.values))  
# 시그모이드를 거친 확률(0~1)을 반올림해서 0 또는 1로 분류
# 결과 : 예측결과 :  [0.  0.  1.  1.  0.]
# 해석 : 소득수준이 23, 46, 0인 집안은 주말외식을 못하고 92, 124인 집안은 주말외식을 진행한다.

print()
# 키보드로 소득 수준(양의 점수)을 입력하면 외식 여부 분류 결과 출력하라.
input_data = int(input('themr tnwns dlqfur:'))
if not isinstance(input_data, int) or input_data < 0:
    print('양의 정수 값을 입력하세요')
else:
    newdf = pd.DataFrame({'소득수준':[input_data]})
    new_pred = result2.predict(newdf)
    if np.around(new_pred.values) == 1:
        print('새 소득수준에 대한 예측 결과 : ', np.around(new_pred.values))
        if np.around(new_pred.values) == 1:
            print('주말 저녁에 외식을 합니다.')
        else:
            print('주말 저녁에 외식을 안합니다.')




