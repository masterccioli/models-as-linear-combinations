# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:54:51 2020

@author: maste
"""

#import modules
import pandas as pd
import numpy as np
import get_wd

import annotated_heat_maps as ahm
import matplotlib.pyplot as plt

def cosine(compare):
    return np.dot(compare,compare.transpose()) / np.outer(np.sqrt(np.sum(compare*compare,1)),np.sqrt(np.sum(compare*compare,1)))

def plot(out,labels):
    # plot
    img_size = 15
    fig, ax = plt.subplots()
    im, cbar = ahm.heatmap(out, labels, labels, ax=ax,
                       cmap="gist_heat", cbarlabel='Jaccard Index')
    ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
    fig.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(img_size, img_size)
    plt.show()
    
data = pd.DataFrame()
#############
# Jaccard first & second order 
wd,mydict = get_wd.loadCorpus('artificialGrammar.txt')

# expected first order
import generalized_jaccard as jac
import pandas as pd

first = jac.get_jaccard_matrix(wd,2,second_order=False)
np.fill_diagonal(first,0)
first = pd.DataFrame(first)
first.columns = mydict.keys()
first.index = mydict.keys()
#first.to_csv('first.csv', index = False)

#plot(first,list(first.columns))

second = jac.get_jaccard_matrix(wd,2,second_order=True)
np.fill_diagonal(second,0)
second = pd.DataFrame(second)
second.columns = mydict.keys()
second.index = mydict.keys()
#second.to_csv('second.csv', index = False)


#plot(second,second.columns)

# convert 1 2 to flattened upper matrix
first = np.array(first)
second = np.array(second)

data['first'] = first[np.triu_indices(first.shape[0],k=1)]
data['second'] = second[np.triu_indices(first.shape[0],k=1)]
#x = np.vstack([first[np.triu_indices(first.shape[0],k=1)],second[np.triu_indices(second.shape[0],k=1)]]).transpose()

##########
# Glove
vects = pd.read_csv('GloVe/vectors/artificialGrammar.txt.txt', header = None, sep = ' ')
vects = vects[:-1]
vects = vects.sort_values(by=[0])
words = sorted(list(vects.loc[:,vects.columns == 0][0]))
out = pd.DataFrame(cosine(vects.loc[:, vects.columns != 0]))
#out.to_csv('glove.csv', index = False)

#plot(out,out.columns)

out = np.array(out)
out = out[np.triu_indices(out.shape[0],k=1)].reshape(66,1)
data['glove'] = out

############
# CBOW

from gensim.models import Word2Vec
import pandas as pd
from os import listdir
import random

def train_model_get_cosine_matrix(statements):
    statements = [statement.split() for statement in statements]
    words = sorted(set([word for statement in statements for word in statement]))
    
    w2v = Word2Vec(statements, size=300, window=3, min_count=1, workers=4, iter=100, sg = 0)
    
    #turn dictionary into doc2vec
    sim = [[w2v.wv.n_similarity([worda],[wordb])
        for wordb in words]
        for worda in words]
    
    out = pd.DataFrame(sim)
    out.columns = words
    out.index = words
    return out

path_out = 'artificialGrammar_.txt'
with open(path_out,'r') as file:
    statements = file.read().split('\n')
random.shuffle(statements)
out = train_model_get_cosine_matrix(statements)
#out.to_csv('cbow.csv', index = False)

#plot(out,out.columns)

out = np.array(out)
out = out[np.triu_indices(out.shape[0],k=1)].reshape(66,1)
data['cbow'] = out


############
# skip-gram

from gensim.models import Word2Vec
import pandas as pd
from os import listdir
import random

def train_model_get_cosine_matrix(statements):
    statements = [statement.split() for statement in statements]
    words = sorted(set([word for statement in statements for word in statement]))
    
    w2v = Word2Vec(statements, size=300, window=3, min_count=1, workers=4, iter=100, sg = 1)
    
    #turn dictionary into doc2vec
    sim = [[w2v.wv.n_similarity([worda],[wordb])
        for wordb in words]
        for worda in words]
    
    out = pd.DataFrame(sim)
    out.columns = words
    out.index = words
    return out

path_out = 'artificialGrammar_.txt'
with open(path_out,'r') as file:
    statements = file.read().split('\n')
random.shuffle(statements)
out = train_model_get_cosine_matrix(statements)
#out.to_csv('skipgram.csv', index = False)
#plot(out,out.columns)

out = np.array(out)
out = out[np.triu_indices(out.shape[0],k=1)].reshape(66,1)
data['skipgram'] = out

#########
# LSA

from gensim import models,corpora
from gensim.similarities import MatrixSimilarity
import pandas as pd
from os import listdir
import numpy as np

def train_model_get_cosine_matrix(statements, num):
    
    statements = [statement.split() for statement in statements]
    dictionary = corpora.Dictionary(statements)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in statements]
    
    ###tfidf model
    # https://stackoverflow.com/questions/50521304/why-i-get-different-length-of-vectors-using-gensim-lsi-model
    tfidf = models.TfidfModel(doc_term_matrix, normalize = True)
    corpus_tfidf = tfidf[doc_term_matrix]
    
    lsi = models.LsiModel(corpus_tfidf, num_topics=num, id2word=dictionary)
    
    #turn dictionary into doc2vec
    words = [dictionary.doc2bow([word])for word in sorted(list(dictionary.token2id.keys()))]
    
    vectorized_corpus = lsi[words]
    
    index = MatrixSimilarity(vectorized_corpus)
    index[vectorized_corpus]
    
    out = pd.DataFrame(index[vectorized_corpus])
    out.columns = sorted(list(dictionary.token2id.keys()))
    out.index = sorted(list(dictionary.token2id.keys()))
    return out

path_out = 'artificialGrammar.txt'
with open(path_out,'r') as file:
    statements = file.read().split('\n')
    
out = train_model_get_cosine_matrix(statements,6)
#out.to_csv('lsa.csv', index = False)
#plot(out,out.columns)

out = np.array(out)
out = out[np.triu_indices(out.shape[0],k=1)].reshape(66,1)
data['lsa'] = out

#######
# PMI

from gensim.models.phrases import npmi_scorer
import pickle
from nltk import FreqDist, ConditionalFreqDist
from os import listdir
import numpy as np
import pandas as pd

def train_model_get_cosine_matrix(statements):
    statements = [statement.split() for statement in statements]

    frequencies = FreqDist(w for word in statements for w in word)

    conditionalFrequencies = ConditionalFreqDist(
                                (key,word)
                                for key in sorted(frequencies.keys())
                                for statement in statements
                                for word in statement 
                                if key in statement)
        
    pmi = [[npmi_scorer(frequencies[worda], 
                  frequencies[wordb], 
                  conditionalFrequencies[worda][wordb], 
                  len(frequencies.keys()),
                  2,
                  sum(frequencies[key] for key in frequencies.keys()))
        for wordb in sorted(frequencies.keys())]
        for worda in sorted(frequencies.keys())]
        
        
    pmi = np.array(pmi)
    pmi[np.isinf(pmi)] = -1
    pmi[np.where(pmi < 0)] = 0
        
    pmi = pd.DataFrame(pmi)
    pmi.columns = sorted(frequencies.keys())
    pmi.index = sorted(frequencies.keys())

    return pmi

path_out = 'artificialGrammar.txt'
with open(path_out,'r') as file:
    statements = file.readlines()
out = train_model_get_cosine_matrix(statements)
#out.to_csv('pmi.csv', index = False)
#plot(out,out.columns)

out = np.array(out)
out = out[np.triu_indices(out.shape[0],k=1)].reshape(66,1)
data['pmi'] = out

data.to_csv('data.csv',index=False)
