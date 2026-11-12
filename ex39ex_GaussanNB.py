# [GaussanNB 문제] 
# 독버섯(poisonous)인지 식용버섯(edible)인지 분류
# https://www.kaggle.com/datasets/uciml/mushroom-classification
# feature는 중요변수를 찾아 선택, label:class
# 참고 : from xgboost import plot_importance

# 데이터 변수 설명 : 총 23개 변수가 사용됨.

# 여기서 종속변수(반응변수)는 class 이고 나머지 22개는 모두 입력변수(설명변수, 예측변수, 독립변수).
# 변수명 변수 설명
# class      edible = e, poisonous = p
# cap-shape    bell = b, conical = c, convex = x, flat = f, knobbed = k, sunken = s
# cap-surface  fibrous = f, grooves = g, scaly = y, smooth = s
# cap-color     brown = n, buff = b, cinnamon = c, gray = g, green = r, pink = p, purple = u, red = e, white = w, yellow = y
# bruises        bruises = t, no = f
# odor            almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s
# gill-attachment attached = a, descending = d, free = f, notched = n
# gill-spacing close = c, crowded = w, distant = d
# gill-size       broad = b, narrow = n
# gill-color      black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w, yellow = y
# stalk-shape  enlarging = e, tapering = t
# stalk-root    bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing = ?
# stalk-surface-above-ring fibrous = f, scaly = y, silky = k, smooth = s
# stalk-surface-below-ring fibrous = f, scaly = y, silky = k, smooth = s
# stalk-color-above-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
# stalk-color-below-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o,pink = p, red = e, white = w, yellow = y
# veil-type      partial = p, universal = u
# veil-color     brown = n, orange = o, white = w, yellow = y
# ring-number none = n, one = o, two = t
# ring-type     cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z
# spore-print-color black = k, brown = n, buff = b, chocolate = h, green = r, orange =o, purple = u, white = w, yellow = y
# population abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y
# habitat       grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d

# ex39iris_naive.py의 흐름: 데이터 준비 -> 숫자로 변환 -> 학습 -> 예측/확률
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import koreanize_matplotlib 

df= pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/mushrooms.csv")
print(df.head(2))
print(df.shape)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col]) 
print(df.head(2)) #e가 0이면 식용, p가 1이면 독버섯 

x= df.drop('class', axis=1)
y=df['class']

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(
    x,y,test_size=0.3, random_state=0,stratify=y
)
print(x_train.shape, x_test.shape)

# 중요변수
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=12)
rf_model.fit(x_train, y_train)

importances =rf_model.feature_importances_
feat_impo = pd.DataFrame({
    'feature':x.columns, 'importance':importances
}).sort_values(by='importance', ascending=False)
print('feature 중요도')
print(feat_impo)

# 중요도 시각화 
import seaborn as sns
plt.figure(figsize=(8, 5))                     
sns.barplot(x='importance', y='feature', data=feat_impo, orient='h')
plt.xlabel('변수별 중요도 점수')
plt.ylabel('독립 변수')
plt.tight_layout()                            
plt.show()
plt.close()

# 10개 추출 
features = feat_impo['feature'].head(10).tolist()
print('상위 10개 feature :', features)
#['odor', 'gill-size', 'gill-color', 'spore-print-color', 'ring-type', 
# 'population', 'stalk-root', 'stalk-surface-above-ring', 
# 'stalk-surface-below-ring', 'bruises']

# 모델 
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train[features], y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

pred = model.predict(x_test[features])
print('예측 값 :', pred[:10]) #예측 값 : [0 0 0 0 1 1 0 0 0 1]
print('실제 값 :', y_test[:10].values) #실제 값 : [0 0 1 0 1 1 0 0 0 1]
print('분류 정확도 :', accuracy_score(y_test, pred)) #분류 정확도 : 0.8740771123872026
print('confusion_matrix : \n', confusion_matrix(y_test, pred)) 
#confusion_matrix : 
# [[1151  112]
# [ 195  980]]

#교차검증
from sklearn.model_selection import cross_val_score

scores = cross_val_score(GaussianNB(), x[features], y, cv=5)
print(f'교차검증에서 각 fold:{scores}, 평균:{scores.mean()}')
#교차검증에서 각 fold:[0.66092308 0.85230769 0.80369231 0.99753846 0.55849754], 평균:0.7745918150814701

# test data 5개로 확인 
sample = x_test[features][:5]
pred1 = model.predict(sample)
prob1 = model.predict_proba(sample)

class_name = ['식용', '독버섯']
for pred_val, real_val, prob in zip(pred1, y_test[:5].values, prob1):
    print(f'예측:{class_name[pred_val]}, 실제:{class_name[real_val]}'
        f'확률:[식용:{prob[0]:.4f}, 독:{prob[1]:.4f}]')