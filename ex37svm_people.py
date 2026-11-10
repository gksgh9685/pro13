# svm 모델로 이미지 분류
# 세계 정치인들 일부 사진 데이터를 이용
# 전체 규모 : 총 5749명의 인물에 대한 13233개의 얼굴 이미지로 구성되어 있다.
# 주요 용도: 얼굴 인식, 이미지 분류, 주성분 분석(pca)등 머신러닝 학습에 쓰인다.
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

faces = fetch_lfw_people(min_faces_per_person=60, color=False, resize=0.5)
# 60 : 한 사람당 60장 이상의 사진이 있는 자료만 사용, 흑백, 크기는 절반
# print(faces.DESCR)
# print(faces.data)
print(faces.data.shape) # (1348, 2914)
print(faces.target)       # [1 3 3 ... 7 3 5]
print(faces.target_names) # ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush'
print(faces.images.shape) #(1348,62,47)
print()
print(faces.images[1])
print(faces.target_names[faces.target[1]])

plt.imshow(faces.images[1], cmap='bone') # 원본 이미지 한 개 시각화
plt.show()

# 원본 이미지 15개 시각화
fig, ax = plt.subplots(3,5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], # x,y 축 숨기기
            xlabel=faces.target_names[faces.target[i]])
plt.show()

# 주성분 분석으로 이미지 차원을 축소시켜 분류 작업을 진행 ---
pca = PCA(n_components=0.95)    # 설명력 95% 이상을 설명할 수 있는 주성분 개수를 선택
x_pca = pca.fit_transform(faces.data)
print(pca.n_components_)

n = 150 # 차원수는 분석가사 결과를 보고 판단함
m_pca = PCA(n_components=n, whiten=True, random_state=0)
# whiten=true : 주성분의 스케일이 작아지도록
x_low = m_pca.fit_transform(faces.data)
print('x_low :', x_low, ' ', x_low.shape)

# 주성분 처리된 이미지 15개 시각화
fig, ax = plt.subplots(3,5, figsize=(10,6))
for i, axi in enumerate(ax.flat):
    axi.imshow(m_pca.components_[i].reshape(faces.images[0].shape), cmap='bone')
    # reshape(faces.images[0].shape) : [2914] -> [62, 47]
    axi.axis('off')
    axi.set_title(f'pc { i + 1}')
plt.suptitle('주성분 얼굴(eigenfaces)', fontsize=12)
plt.tight_layout()
plt.show()  # 출력 이미지는 실제가 아니라 특징 패턴(얼굴 윤곽, 코그림자, 눈위치...)을 보여줌
# svm 알고리즘은 실제 얼굴이 아니라 특징 패턴으로 학습한다.

print('주성분 설명력 : \n', m_pca.explained_variance_ratio_[:10])
print('누적 설명력 :', m_pca.explained_variance_ratio_.sum()) # 누적 설명력 : 0.93590003 : n = 100개로 얼마나 원본 정보를 유지했는지 확인
# 주성분 설명력 :  [0.20795499 0.13552667 0.06788544 0.05741937 0.04768271 0.0276972 0.0234214  0.01973507 0.01856109 0.01784675]

# 원본 vs 복원 이미지 비교
x_reconst = m_pca.inverse_transform(x_low)
fig, ax = plt.subplots(2, 5, figsize=(10,4))
for i in range(5):
    # 원본
    ax[0, i].imshow(faces.images[i], cmap='bone')
    ax[0,i].set_title('원본')
    ax[0,i].axis('off')

    # 복원
    ax[1,i].imshow(x_reconst[i].reshape(faces.images[0].shape), cmap='bone')
    ax[1,i].set_title('복원')
    ax[1,i].axis('off')
plt.suptitle('pca 복원 비교', fontsize=12)
plt.tight_layout()
plt.show()  # 원본과 복원 이미지의 기본 특징은 크게 차이가 없다.(패턴이 계속 유지)

print()
svcmodel = SVC(C=1.0, random_state=1)
mymodel = make_pipeline(m_pca, svcmodel)    # 
print('mymodel : ', mymodel)
# steps=[('pca'), PCA(...)), ('svc',svc(random_state=1))]

# train / test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    faces.data, faces.target, random_state=1, stratify=faces.target
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print(x_train[0])
print(y_train[0])

mymodel.fit(x_train, y_train)
pred = mymodel.predict(x_test)
print('예측값:',pred[:10])
print('실제값:',y_test[:10])

# 정확도
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confmat = confusion_matrix(y_test, pred)
print('confusion_matrix : \n', confmat)
print('accuracy :', accuracy_score(y_test, pred))
print('classfication_report : \n',classification_report(y_test, pred, target_names = faces.target_names))

print('분류 결과 시각화')   # 여러 개 보기
print('분류 결과 시각화')    # 여러 개 보기
fig, axes = plt.subplots(4, 6)
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape(62, 47), cmap='bone')
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel(faces.target_names[pred[i]].split()[-1], \
                color='blue' if pred[i] == y_test[i] else 'red',
                fontweight='bold')

fig.suptitle('예측 결과', fontsize=12)
plt.tight_layout()
plt.show()

# 오차 행렬 시각화 (heatmap)
import seaborn as sns
sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues',
            xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('예측')
plt.ylabel('실제')
plt.title('혼동행렬')
plt.show()

print('\n새로운 이미지를 입력해 분류하기 ------------')
# 실습 1 : ㅣㄱ존 이미지로 분류 예측
test_img = faces.data[0].reshape(1,-1)  #
print('test_img :', test_img)
test_pred = mymodel.predict(test_img)
print('실습1 예측 결과:', faces.target_names[test_pred[0]],', index:',test_pred[0])
print('실제값 : ', faces.target_names[faces.target[0]],', index:', faces.target[0])
# 실습1 예측 결과: Colin Powell , index: 1
# 실제값 :  Colin Powell , index: 1

# 실습2 : 완전 새로운 이미지로 분류 예측 (이게 하고 싶은 작업)
# 단계 : 이미지 읽기 -> 흑백변환 -> 크기 맞추기(62 * 47) -> 1차원으로 변환 -> 예측
from PIL import Image
import numpy as np

img = Image.open('bush.jpg')    # 이미지 읽기
img = img.convert('L')          # 흑백변환
img = img.resize((47,62))       # 크기 맞추기 (width, height)
# numpy 이미지는 (hight, width) - 세로, 가로
# PIL 이미지는 (width, height) - 가로, 세로, 이미지는 라이브러리 마다 축 순서가 다를 수 있다.
img_np = np.array(img)  # numpy 배열로 변환
print('img_np : ', img_np)  # [[48 48 19 ... 62 62 62]...
img_np = img_np /255.0 # 정규화 (학습 데이터도 이렇게 되어 있음)
print('img_np :', img_np)   # img_np : [[0.18823529 0.18823529 0.0745098  ... 0.24313725 0.24313725 0.24313725]

img_flat = img_np.reshape(1, -1)    # 

# 예측
new_pred=mymodel.predict(img_flat)
print('실습2 예측 결과 : ', faces.target_names[new_pred[0]], ', index', new_pred[0])
# 실습2 예측 결과 :  George W Bush , index 3

# 시각화 + 예측
plt.imshow(img_np, cmap='bone')
plt.title(f'예측: {faces.target_names[new_pred[0]]}')
plt.axis('off')
plt.show()

