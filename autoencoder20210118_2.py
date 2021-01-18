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
# split the data into a training set and a validation set
random.seed(9999)
VALIDATION_SPLIT = 0.2

data = np.array(data)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
x_val = data[-num_validation_samples:]
len(x_train)
len(x_val)


from keras.layers import Input, Dense
from keras.models import Model



# intput size
input_dim = 300

# this is the size of our encoded representations
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(200, activation='tanh')(input_img)
encoded = Dense(100, activation='tanh')(encoded)
encoded = Dense(50, activation='tanh')(encoded)
encoded = Dense(encoding_dim, activation='tanh')(encoded)

# "decoded" is the lossy reconstruction of the input
#decoded = Dense(encoding_dim, activation='sigmoid')(encoded)
decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)
decoded = Dense(200, activation='tanh')(decoded)
decoded = Dense(input_dim, activation='tanh')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.summary()

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input01
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-4]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# compile
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

## train
history = autoencoder.fit(x_train, x_train,
                          epochs=2000,
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


def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

plot_acc(history)
plot_loss(history)

#tanh epoch 200 = 7.5

## test
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_val)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs.shape)
print('z: ' + str(encoded_imgs))


import math
def my_rmse(np_arr1,np_arr2):
    dim = np_arr1.shape
    tot_loss = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            tot_loss += math.pow((np_arr1[i,j] - np_arr2[i,j]),2)
    return round(math.sqrt(tot_loss/(dim[0]* dim[1]*1.0)),2)

my_rmse(x_val, decoded_imgs)

#ipc에 따른 카테고리 필요 (labeling) - done!
#해당 카테고리에 대해서 Doc2vec 수행 - done!
#Doc2vec 데이터에 대해서 AE 모델 훈련 - done!
