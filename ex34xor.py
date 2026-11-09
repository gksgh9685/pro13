# xor 분류 : logisticregression과 svm으로 분류

x_data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,1],
]

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics

x_df = pd.DataFrame(x_data)
print(x_df)
feature = np.array(x_df.iloc[:,0:2])
label = np.array(x_df.iloc[:,2])
print(feature)
print(label)

lmodel = LogisticRegression()   # 선형 분류 모델
smodel = svm.SVC()  # 선형(비선형)분류 모델

lmodel.fit(feature, label)
smodel.fit(feature, label)

pred1 = lmodel.predict(feature)
print('lmodel pred :', pred1)
pred2 = smodel.predict(feature)
print('smodel pred :', pred2)

acc1 = metrics.accuracy_score(label, pred1)
print('lmodel accuracy : ', acc1)
acc2 = metrics.accuracy_score(label, pred2)
print('smodel accuracy : ', acc2)





