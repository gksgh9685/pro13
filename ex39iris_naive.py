# spam 메일 분류기

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# Multinomial Naive Bayes (다항 나이브 베이즈).
# 텍스트에서 가장 많이 쓰는 NB 모델
# 단어의 “빈도(count)”를 기반으로 분류함
# 형태: “단어가 많이 나왔으면 그 단어 빈도에 비례해 확률이 커짐”

# 학습용 데이터
texts = [
    '무료 쿠폰 지금 클릭하면 무료 선물',
    '한번만 클릭해도 무료 대박',
    '오늘 회의는 2시 제3회의실입니다',
    '지금 할인 행사 중',
    '회의 자료는 메일로 보내주세요',
    '지금 바로 쿠폰 확인'
]
labels = ['spam','spam','ham','spam','ham','spam']
# 단어 등장 횟수 비잔 벡터 생성
vect = CountVectorizer()    # 문서로부터 단어의 순서정보는 무시하고, 단어 빈도수 정보를 반환
x = vect.fit_transform(texts)
print(vect.get_feature_names_out()) # ['2시' '대박' '메일로' '무료' '바로' '보내주세요' ..
print(x.toarray())  # [[0 0 0 2 0 ...]]
print(vect.vocabulary_) # {'무료': 3, '쿠폰': 12, '지금': 11, ...
print(x)    #   (0, 3)        2 <-- (문서 번호, 단어 번호) 등장횟수

# 모델
from sklearn.metrics import accuracy_score
model = MultinomialNB()
model.fit(x, labels)
pred = model.predict(x)
print('정확도 :', accuracy_score(labels, pred)) # 1.0

print()
# 새로운 문장으로 분류
text_test =['무료 쿠폰 지금 발급','간부 회의는 언제 시작하나요']
x_test = vect.transform(text_test)
print(x_test)

# 예측 + 확률 출력
preds = model.predict(x_test)
probs = model.predict_proba(x_test)
class_name = model.classes_

for text, pred, prob in zip(text_test, preds, probs):
    prob_str=','.join([f'{cls}:{p:.4f}' for cls, p in zip(class_name, prob)])
    print(f"'{text}' -> 예측:{pred} // 확률:[{prob_str}]")

# '무료 쿠폰 지금 발급' -> 예측:spam // 확률:[ham:0.0235,spam:0.9765]
# '간부 회의는 언제 시작하나요' -> 예측:ham // 확률:[ham:0.5692,spam:0.4308]