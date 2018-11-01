import gensim
from os import listdir
from os.path import isfile, join
import json
import numpy as np

data = ""
count = 0
keys = {}


#data_store_bola
topic = "bola"
dir_path = topic+"/"
docLabels = [f for f in listdir(dir_path) if
             f.endswith(".json")]

for doc in docLabels[:int(len(docLabels)/4)]:
    jdata = json.load(open(dir_path+doc))
    for j in jdata:
        for key, value in j.items():
            data+=value
            data+="\n"
            keys[count] = key
            count+=1

#data_store_tekno
topic = "tekno"
dir_path = topic+"/"
docLabels = [f for f in listdir(dir_path) if
             f.endswith(".json")]

for doc in docLabels[:int(len(docLabels)/4)]:
    jdata = json.load(open(dir_path+doc))
    for j in jdata:
        for key, value in j.items():
            data+=value
            data+="\n"
            keys[count] = key
            count+=1

#data_store_bisniskeuangan
topic = "bisniskeuangan"
dir_path = topic+"/"
docLabels = [f for f in listdir(dir_path) if
             f.endswith(".json")]

for doc in docLabels[:int(len(docLabels)/4)]:
    jdata = json.load(open(dir_path+doc))
    for j in jdata:
        for key, value in j.items():
            data+=value
            data+="\n"
            keys[count] = key
            count+=1

file = open("doc2vec_data.txt", "w")
file.write(data)
file.close()

#save keys
np.save("keys", keys)

sent = gensim.models.doc2vec.TaggedLineDocument("doc2vec_data")
model = gensim.models.doc2vec.Doc2Vec(sent, size=300,window=10,min_count=5,
                                      iter=20, workers=32)
model.save("doc2vec_model.d2v")

    
