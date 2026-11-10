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
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

# 1. 학습용 데이터: 같은 폴더의 mushrooms.csv가 있으면 사용한다.
# 없으면 UCI 원본 데이터를 읽는다. 최초 실행에는 인터넷 연결이 필요하다.
csv_path = Path(__file__).with_name('mushrooms.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat',
    ]
    url = 'https://archive.ics.uci.edu/static/public/73/mushroom.zip'
    with urlopen(url, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        with archive.open('agaricus-lepiota.data') as source:
            df = pd.read_csv(source, header=None, names=columns)

print('데이터 크기:', df.shape)
x = df.drop('class', axis=1)
labels = df['class']
# stalk-root의 '?'는 '알 수 없음'이라는 별도 범주로 처리한다.
x_train, x_test, y_train, y_test = train_test_split(
    x, labels, test_size=0.2, random_state=42, stratify=labels
)

# 2. 중요변수 선택: 범주에 임의의 숫자 순서를 부여하지 않고 정보를 비교한다.
# 훈련 데이터만 사용해 정답 정보가 테스트에서 유입되지 않도록 한다.
x_codes = x_train.apply(lambda col: pd.factorize(col)[0])
selector = SelectKBest(
    score_func=lambda data, target: mutual_info_classif(
        data, target, discrete_features=True, random_state=42
    ),
    k=5,
)
selector.fit(x_codes, y_train)
feat_impo = pd.DataFrame({
    'feature': x.columns, 'importance': selector.scores_,
}).sort_values('importance', ascending=False)
print('변수 중요도(상호정보량):\n', feat_impo)
features = x.columns[selector.get_support()].tolist()
print('선택한 중요변수:', features)

# 3. ex39의 CountVectorizer 대신 범주형 데이터용 OneHotEncoder 사용
# GaussianNB는 밀집 배열을 입력받으므로 sparse_output=False로 설정한다.
vect = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_data = vect.fit_transform(x_train[features])
test_data = vect.transform(x_test[features])

# 4. 모델 학습: 문제에서 요구한 GaussianNB 사용
# 버섯 특성은 범주형이므로 정규분포 가정에는 한계가 있다.
model = GaussianNB()
model.fit(train_data, y_train)

# 5. 학습에 사용하지 않은 데이터로 평가
pred = model.predict(test_data)
print('분류 정확도:', accuracy_score(y_test, pred))
print('혼동행렬 (행: 실제, 열: 예측 / 순서: e, p):\n',
      confusion_matrix(y_test, pred, labels=['e', 'p']))

# 6. ex39와 같은 방식으로 테스트 자료 5개의 예측 + 확률 출력
newdata = x_test[features].head(5)
x_new = vect.transform(newdata)
preds = model.predict(x_new)
probs = model.predict_proba(x_new)
class_name = model.classes_
names = {'e': '식용', 'p': '독버섯'}
for index, pred, prob in zip(newdata.index, preds, probs):
    prob_str = ','.join(
        f'{names[cls]}:{p:.4f}' for cls, p in zip(class_name, prob)
    )
    print(f'{index}번 -> 실제:{names[y_test.loc[index]]} '
          f'예측:{names[pred]} // 확률:[{prob_str}]')






