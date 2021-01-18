#### data import
import pandas as pd


pat3 = pd.read_csv('C:/Users/owner/Documents/research/autoencoder/HTpatent1719.txt', sep='\t', error_bad_lines=False)

pat3.dtypes
pat3.shape
pat3['pnyear'].head()

def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)


which(pat3['pnyear']==2019).__len__()
pat3.__len__()

pat2019 = pat3.iloc[which(pat3['pnyear']==2019),:]
len(pat2019)

pat2019.columns

#test data
pat19kr = pat2019.iloc[which(pat2019['authority']=='KR'),:]

#tr data
pat19us = pat2019.iloc[which(pat2019['authority']=='US'),:]




#### pre-processing
import re
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

stemmer = WordNetLemmatizer()

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

#for preproc test
print(preprocess_text("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))



# for us data
word_punctuation_tokenizer = nltk.WordPunctTokenizer()
corpus = []
for i in range(0,len(pat19us)): #len(pat))
    patData = pat19us.iloc[i,6] + ' ' + pat19us.iloc[i,7]+' '+ pat19us.iloc[i,8].replace(";"," ")
    patDataPrep = preprocess_text(patData)
    patDataTokenCorp = word_punctuation_tokenizer.tokenize(patDataPrep)
    corpus.append(patDataTokenCorp)
    print("iteration:", i,"/",len(pat19us))


len(corpus)==pat19us.shape[0]


# representative IPC
import operator

label01 = ['B41J','G06C','G06D','G06E','G11C','G06Q','G06G','G06J','G06F','G06M']
label02 = ['B64B','B64C','B64D','B64F','B64G']
label03 = ['C40B','C12P', 'C12Q']
label04 = ['H01S']
label05 = ['H01L']
label06 = ['H04B','H04H','H04J','H04K','H04L','H04M','H04N','H04Q','H04R','H04S']


def intersect(a, b):
    return list(set(a) & set(b))

label = []
for j in range(0, len(corpus)): #len(corpus)
    tmp = pat19us.iloc[j, 10].replace(';', ' ')
    tmp = tmp.split(' ')

    match01 = intersect(tmp, label01)
    match02 = intersect(tmp, label02)
    match03 = intersect(tmp, label03)
    match04 = intersect(tmp, label04)
    match05 = intersect(tmp, label05)
    match06 = intersect(tmp, label06)

    tmpOutput = match01+match02+match03+match04+match05+match06

    if len(tmpOutput)>1:
        labelCount = []
        for k in range(0,len(tmpOutput)):
            labelCount.append(tmp.count(tmpOutput[k]))
        labelCountIdx, labelCountVal = max(enumerate(labelCount), key=operator.itemgetter(1))
        tmpOutput = tmpOutput[labelCountIdx]

    elif len(tmpOutput)==1:
        tmpOutput = tmpOutput[0]

    else:
        tmpOutput = 'NA'


    label.append(tmpOutput)
    print("iteration:", j, "/", len(corpus))



# labeling
labelTi = ['computer and automated business equipment', 'aviation', 'micro-organism and genetic engineering','lasers','semiconductors','communication technology','NA']

for l in range(0, len(label)):
    input = label[l]
    if label01.count(input) == 1:
        label[l] = labelTi[0]

    elif label02.count(input) == 1:
        label[l] = labelTi[1]

    elif label03.count(input) == 1:
        label[l] = labelTi[2]

    elif label04.count(input) == 1:
        label[l] = labelTi[3]

    elif label05.count(input) == 1:
        label[l] = labelTi[4]

    elif label06.count(input) == 1:
        label[l] = labelTi[5]

    else:
        label[l] = labelTi[6]

    print("iteration:", l, "/", len(corpus))


print(labelTi[0], label.count(labelTi[0])) #22732
print(labelTi[1], label.count(labelTi[1])) #4291
print(labelTi[2], label.count(labelTi[2])) #5371
print(labelTi[3], label.count(labelTi[3])) #1120
print(labelTi[4], label.count(labelTi[4])) #26523
print(labelTi[5], label.count(labelTi[5])) #48591
print(labelTi[6], label.count(labelTi[6])) #142

len(pat19us)
len(label)


# export
import pickle
import gzip

# save and compress.
with gzip.open('corpusUs.pickle', 'wb') as f:
    pickle.dump(corpus, f)


with gzip.open('label.pickle', 'wb') as f:
    pickle.dump(label, f)


# load and uncompress.
with gzip.open('corpusUs.pickle','rb') as f:
    corpus = pickle.load(f)


with gzip.open('label.pickle','rb') as f:
    label = pickle.load(f)


#ipc에 따른 카테고리 필요 (labeling) - done!
#해당 카테고리에 대해서 Doc2vec 수행
#Doc2vec 데이터에 대해서 AE 모델 훈련


