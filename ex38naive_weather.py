# 나이브베이즈 알고리즘을 이용한 분류 - weather
import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv')
print(df.head(2))
print(df.info())

# 전처리 ------
df = df.drop('Date', axis=1)
df['Sunshine'] = df['Sunshine'].fillna(df['Sunshine'].mean())   # 결측치 처리

# 범주형 처리
df['RainToday'] = df['RainToday'].map({'Yes':1,'No':0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1,'No':0})
print(df.head(2))

x= df.drop('RainTomorrow', axis=1)
y= df['RainTomorrow']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42, stratify=y
)

# naivebayes 모델 학습
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
model = GaussianNB()
model.fit(x_train, y_train)

# 예측 및 평가
from sklearn.metrics import accuracy_score, confusion_matrix
pred = model.predict(x_test)
print('분류 정확도 :', accuracy_score(y_test, pred))
print('confusion_matrix : \n',confusion_matrix(y_test, pred))

print() # 교차 검증
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5)
print(f'교차 검증에서 각 fold:{scores}, 평균{scores.mean()}')   # 교차 검증에서 각 fold:[0.72972973 0.82191781 0.79452055 0.8630137  0.83561644], 평균0.8089596445760829

print()# feature 중요도 분석 - feature가 정규분포를 따른다는 가정하에 클래스별 평균
# gaussianNB의 멤버로 theta_ : 각 클래스별 feature 평균으르 반환
mean_0 = model.theta_[0]    # RainTomorrow=0    (비 안오는 날 평균)
mean_1 = model.theta_[1]    # RainTomorrow=1    (비 오는 날 평균)

# 각 feature가 '비 오는 날 vs 비 안오는 날'에서 얼마나 차이가 나는가?
importance = np.abs(mean_1 - mean_0)

feat_impo = pd.DataFrame({
    'feature':x.columns, 'importance':importance
}).sort_values(by='importance',ascending=False)
print('feature 중요도')
print(feat_impo)

# importanve 시각화
import matplotlib.pyplot as plt
import koreanize_matplotlib
plt.figure()
plt.bar(feat_impo['feature'], feat_impo['importance'])
plt.xlabel('feature')
plt.ylabel('중요도(평균차이)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print('새로운 자료에 대한 예측')
newdata = pd.DataFrame([{
    'MinTemp': 1.3,
    'MaxTemp': 15.0,
    'Rainfall': 0.0,
    'Sunshine': 10,
    'WindSpeed': 3,
    'Humidity': 40,
    'Pressure': 1000,
    'Cloud': 3,
    'Temp': 20,
    'RainToday': 0,
}])
newpred = model.predict(newdata)
print('예측 결과 :', '비옴' if newpred == 1 else '비안옴')
print('확률은 ', model.predict_proba(newdata))

