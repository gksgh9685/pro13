# ROC(Receiver Operating Characteristic) Curve는 모든 분류 
# 임계값에서 분류 모델의 성능을 보여주는 그래프로 x축이 FPR(1-특이도), 
# y축이 TPR(민감도)인 그래프이다.
# 즉 민감도와 특이도의 관계를 표현한 그래프이다.
# ROC Curve는 AUC(Area Under Curve : 그래프 아래의 면적)를 이용해 
# 모델의 성능을 평가한다. AUC가 클수록 정확히 분류함을 뜻한다. 
# fpr(1-특이도 : 위양성률)이 변할 때 tpr(민감도)이 어떻게 변하는지 알려 주는 곡선이다.

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123)
# n_redundant : 독립변수 중 다른 독립변수의 선형조합으로 나타내는 성분 수
print(x[:3], x.shape)
print(y[:3], y.shape)

# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

model = LogisticRegression().fit(x, y)
y_hat = model.predict(x)
print('y_hat : ', y_hat[:10])   # [0 0 0 1 1 1 0 0 0 1]
print('real : ', y[:10])        # [1 0 0 1 0 1 0 1 0 1]

# ROC Curve의 판별 경계선 설정용 결정함수로 데이터
f_value = model.decision_function(x)
print('f_value : ', f_value[:10])
df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns=['f', 'y_hat', 'y'])
print(df.head())

# 모델 성능 파악
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_hat))
acc = (44 + 44 ) / 100       # TP + TN / 전체수   정확도
recall = 44 / (44 + 4)       # TP / (TP + FN)    재현율
precision = 44 / (44 + 8)    # TP / (TP + FP)  정밀도
specificity = 44 / (8 + 44)  # TN / (FP + TN) 특이도
fallout = 8 / (8 + 44)       # FP / (FP + TN)  위양성률 (fpr)
print('acc : ', acc)
print('recall : ', recall)
print('precision : ', precision)
print('specificity : ', specificity)
print('fallout : ', fallout)
print('fallout : ', 1 - specificity)

print()
from sklearn import metrics
acc_score = metrics.accuracy_score(y, y_hat)
print('acc(정확도) :', acc_score)

cl_rep = metrics.classification_report(y, y_hat)
print('분류모델 성능 평가 보고서 : \n', cl_rep)

# ROC Curve의 판별 경계선 설정용 결정함수로 데이터 - 참고
f_value = model.decision_function(x)
print('f_value : ', f_value[:10])
df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns=['f', 'y_hat', 'y'])
print(df.head())
# ----

fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
# thresholds : 분류 결정 임계값 (결정수값)

plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0, 1], [0, 1], 'k--', label='landom classifier line(AUC:0.5)')
# plt.plot([fallout], [recall], 'ro', ms=6)   # 위양성률, 재현율 출력
# 위 명령 대신 : 특정 임계값에 해당하는 ROC 좌표로 표시
i=3
plt.plot([fpr[i]], [tpr[i]], 'ro', ms=6, label='선택한 임계값')

plt.xlabel('fpr(위양성률)')
plt.xlabel('tpr(재현율)')
plt.title('ROC Curve')
plt.legend()
plt.show()

print('auc(area under the curve): roc 밑면적의, 1에 근사할 수록 좋은 모양')
print('auc : ', metrics.auc(fpr,tpr))   # auc :  0.9547275641






