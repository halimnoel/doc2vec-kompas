from gensim.models import Doc2Vec
import os
import numpy as np
import random
model = Doc2Vec.load("doc2vec_model.d2v")

#load keys
keys = np.load("keys.npy").item()
count_true = 0


idx = random.sample(range(0,18189),3000)
for i in idx:
    sims = model.docvecs.most_similar(i)

    kt = keys[i][8]
    #bola 0-5536
    #tekno 5537-10245
    #eknomi 10246-18190


    for sim in sims:
        if (keys[sim[0]][8] == kt):count_true+=1

print(count_true)
print("accuracy: ", (count_true/30000)*100)

# bola 79.66
# tekno 78.7
#ekonomi 79.12
#clustering performance evaluation
