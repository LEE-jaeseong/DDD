from keras.models import model_from_json

# 모델 불러오기
json_file = open("autoencoderELU200.json", "r")
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)

# 모델 가중치 불러오기
autoencoder.load_weights("autoencoderELU200.h5")
print("Loaded model from disk")

# 모델을 사용할 때는 반드시 컴파일을 다시 해줘야 한다.
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
score = autoencoder.evaluate(x_val, x_val,verbose=0)




# Threshold 구하기
import numpy as np
import math

np.random.seed(42)
data = np.random.rand(800, 4) # change data -> encoded data

def mahalanobis(x, y, cov=None):  #y 전체, x 일부
    x_mean = np.mean(x)
    Covariance = np.cov(np.transpose(y))
    inv_covmat = np.linalg.inv(Covariance)
    x_minus_mn = x - x_mean
    D_square = np.dot(np.dot(x_minus_mn, inv_covmat), np.transpose(x_minus_mn))
    return math.sqrt(D_square)


mahalanobis(data[0], data)

tmp = []
for i in range(0, len(data)):
    print('iteration {0}'.format(i))
    tmp.append(mahalanobis(data[i], data))


np.sort(tmp)[::-1]
Threshold = np.sort(tmp)[::-1][int(len(tmp) * 0.95)]



import umap
import matplotlib.pyplot as plt

fit = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, metric='mahalanobis') # neighbors number = 라벨 개수
u = fit.fit_transform(data)
plt.scatter(u[:,0], u[:,1], c=data)
plt.title('UMAP embedding of random colours')
# https://umap-learn.readthedocs.io/en/latest/parameters.html


import pandas as pd
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=Threshold, min_samples=Threshold, metric='mahalanobis', metric_params={'V':np.cov(data)}, algorithm='brute', leaf_size=30, n_jobs=-1) # Threshold of PPDM
predict = pd.DataFrame(model.fit_predict(u)) # chain with umap
predict.columns=['predict']

r = pd.concat([pd.DataFrame(data),predict],axis=1)
r['predict'].describe()
pd.Series.unique(r['predict'])
r['predict'].value_counts()

plt.scatter(u[:,0], u[:,1], c=r['predict'])





