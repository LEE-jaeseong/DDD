from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import pickle
import gzip
import pandas as pd

# load and uncompress.
with gzip.open('label.pickle','rb') as f:
    label = pickle.load(f)


#https://wjddyd66.github.io/tensorflow/Tensorflow-AutoEncoder/#stacked-autoencoder%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%B9%84%EC%A7%80%EB%8F%84-%EC%82%AC%EC%A0%84%ED%95%99%EC%8A%B5%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5
#https://deepinsight.tistory.com/126

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)


# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('103241')
print(similar_doc)

model.corpus_count
model.corpus_total_words
model.wv.vectors.shape
model.wv.vocab
len(model.docvecs)


data = []
for i in range(0, len(model.docvecs)):
    print('iteration {0}'.format(i))
    data.append(model.docvecs[i])


import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
# split the data into a training set and a validation set
random.seed(9999)
VALIDATION_SPLIT = 0.2

data = np.array(data)
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(data)
# min_max_scaler.transform(newData)
# min_max_scaler.inverse_transform(data)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
x_val = data[-num_validation_samples:]
len(x_train)
len(x_val)


from keras.layers import Input, Dense, LeakyReLU, ELU
from keras.models import Model

# intput size
input_dim = 300

# this is the size of our encoded representations
encoding_dim = 50

# 01 relu model, decoder layer = -5, eval = 0.0081
# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(250, activation='relu')(input_img)
encoded = Dense(200, activation='relu')(encoded)
encoded = Dense(150, activation='relu')(encoded)
encoded = Dense(100, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
#decoded = Dense(encoding_dim, activation='sigmoid')(encoded)
decoded = Dense(100, activation='relu')(encoded)
decoded = Dense(150, activation='relu')(decoded)
decoded = Dense(200, activation='relu')(decoded)
decoded = Dense(250, activation='relu')(decoded)
decoded = Dense(input_dim, activation='relu')(decoded)


# 02 LeakyReLU model, decoder layer = -10, eval = 0.0063
# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(250)(input_img)
encoded = LeakyReLU(alpha=0.01)(encoded)
encoded = Dense(200)(encoded)
encoded = LeakyReLU(alpha=0.01)(encoded)
encoded = Dense(150)(encoded)
encoded = LeakyReLU(alpha=0.01)(encoded)
encoded = Dense(100)(encoded)
encoded = LeakyReLU(alpha=0.01)(encoded)
encoded = Dense(encoding_dim)(encoded)
encoded = LeakyReLU(alpha=0.01)(encoded)

# "decoded" is the lossy reconstruction of the input
#decoded = Dense(encoding_dim, activation='sigmoid')(encoded)
decoded = Dense(100)(encoded)
decoded = LeakyReLU(alpha=0.01)(decoded)
decoded = Dense(150)(decoded)
decoded = LeakyReLU(alpha=0.01)(decoded)
decoded = Dense(200)(decoded)
decoded = LeakyReLU(alpha=0.01)(decoded)
decoded = Dense(250)(decoded)
decoded = LeakyReLU(alpha=0.01)(decoded)
decoded = Dense(input_dim)(decoded)
decoded = LeakyReLU(alpha=0.01)(decoded)


# 03 ELU model, decoder layer = -10, eval = 0.0068, rmse = 0.12, rmsle = 0.08
# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(250)(input_img)
encoded = ELU(alpha=1)(encoded)
encoded = Dense(200)(encoded)
encoded = ELU(alpha=1)(encoded)
encoded = Dense(150)(encoded)
encoded = ELU(alpha=1)(encoded)
encoded = Dense(100)(encoded)
encoded = ELU(alpha=1)(encoded)
encoded = Dense(encoding_dim)(encoded)
encoded = ELU(alpha=1)(encoded)

# "decoded" is the lossy reconstruction of the input
#decoded = Dense(encoding_dim, activation='sigmoid')(encoded)
decoded = Dense(100)(encoded)
decoded = ELU(alpha=1)(decoded)
decoded = Dense(150)(decoded)
decoded = ELU(alpha=1)(decoded)
decoded = Dense(200)(decoded)
decoded = ELU(alpha=1)(decoded)
decoded = Dense(250)(decoded)
decoded = ELU(alpha=1)(decoded)
decoded = Dense(input_dim)(decoded)
decoded = ELU(alpha=1)(decoded)


# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.summary()

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input01
encoded_input = Input(shape=(encoding_dim,))

# deco = autoencoder.layers[-1](encoded_input)
deco = autoencoder.layers[-10](encoded_input)
deco = autoencoder.layers[-8](deco)
deco = autoencoder.layers[-6](deco)
deco = autoencoder.layers[-4](deco)
deco = autoencoder.layers[-2](deco)


# create the decoder model
decoder = Model(encoded_input, deco)

# compile
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

## train
history = autoencoder.fit(x_train, x_train,
                          epochs=200,
                          batch_size=64,
                          shuffle=True,
                          validation_data=(x_val, x_val))


import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)


#def plot_acc(history):
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('Model accuracy')
#    plt.ylabel('Accuracy')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'Test'], loc=0)

#plot_acc(history)
plot_loss(history)

## test
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_val)
decoded_imgs = decoder.predict(encoded_imgs)

print(x_val.shape)
print(encoded_imgs.shape)
print(decoded_imgs.shape)
print('z: ' + str(encoded_imgs))


import math
def my_rmse(np_arr1, np_arr2):
    dim = np_arr1.shape
    tot_loss = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            tot_loss += math.pow((np_arr1[i, j] - np_arr2[i, j]), 2)
    return round(math.sqrt(tot_loss/(dim[0] * dim[1]*1.0)),2)


def rmsle(predicted_values, actual_values):
    # 넘파이로 배열 형태로 바꿔줌.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    # 예측값과 실제 값에 1을 더하고 로그를 씌어줌
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)

    # 위에서 계산한 예측값에서 실측값을 빼주고 제곱해줌
    difference = log_predict - log_actual
    difference = np.square(difference)

    # 평균을 냄
    mean_difference = difference.mean()

    # 다시 루트를 씌움
    score = np.sqrt(mean_difference)

    return score


my_rmse(x_val, decoded_imgs)
rmsle(decoded_imgs, x_val)


print(x_val[0])
print(decoded_imgs[0])



# save the model and weights

autoencoder_json = autoencoder.to_json()
with open("autoencoder.json", "w") as json_file :
    json_file.write(autoencoder_json)

autoencoder.save_weights("autoencoderELU200.h5")
print("Saved model to disk")

encoder_json = encoder.to_json()
with open("encoder.json", "w") as json_file :
    json_file.write(encoder_json)

encoder.save_weights("encoderELU200.h5")
print("Saved model to disk")

decoder_json = decoder.to_json()
with open("decoder.json", "w") as json_file :
    json_file.write(decoder_json)

decoder.save_weights("decoderELU200.h5")
print("Saved model to disk")




#ipc에 따른 카테고리 필요 (labeling) - done!
#해당 카테고리에 대해서 Doc2vec 수행 - done!
#Doc2vec 데이터에 대해서 AE 모델 훈련 - done!
#https://medium.com/rapids-ai/tsne-with-gpus-hours-to-seconds-9d9c17c941db
#https://otexts.com/fppkr/accuracy.html

