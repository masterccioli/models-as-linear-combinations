# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:00:45 2020

@author: maste
"""

import numpy as np
from scipy.signal import fftconvolve,deconvolve
import get_wd
from scipy import sparse

import annotated_heat_maps as ahm
import matplotlib.pyplot as plt


# generate random vectors for each word
dims = 10000
wd,mydict = get_wd.loadCorpus('../../first_second_order/corpus/artificialGrammar.txt')
vects = np.random.normal(0,1/np.sqrt(dims),(len(mydict),dims))

def cosine_table_self(vects): # get cosine table, input one matrix
    return vects.dot(vects.transpose()) / \
            np.outer(np.sqrt(np.power(vects, 2).sum(1)),
                     np.sqrt(np.power(vects,2).sum(1)))
            
        
def cosine_table(vects_a,vects_b): # get cosine sims between two matrices
    return vects_a.dot(vects_b.transpose()) / \
            np.outer(np.sqrt(np.power(vects_a,2).sum(1)),
                     np.sqrt(np.power(vects_b,2).sum(1)))

cosine_table_self(vects)

# input layer is a localist word activation
# output layer is a distributed activation
# where the distribution is the sum of convolved vectors where a word occurs
mem_conv = np.zeros((len(mydict),dims),'complex128')
for row in wd:
    print(sparse.find(row)[1])
    vect = np.ones(dims,'complex128')
    for i in sparse.find(row)[1]:
        vect *= np.fft.fft(vects[i])
    for i in sparse.find(row)[1]:
        mem_conv[i] += vect

mem_conv = np.fft.irfft(mem_conv)

first_conv = cosine_table_self(mem_conv)
np.fill_diagonal(first_conv,0)

img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(first_conv, sorted(mydict.keys()), sorted(mydict.keys()), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)

second_conv = cosine_table_self(first_conv)
np.fill_diagonal(second_conv,0)

img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(second_conv, sorted(mydict.keys()), sorted(mydict.keys()), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)
#plt.show()
#fig.savefig('heatmaps/glove.png', dpi=100, transparent = True)

mem_sum = np.zeros((len(mydict),dims))
for row in wd:
    print(sparse.find(row)[1])
    for i in sparse.find(row)[1]:
        mem_sum[i] += vects[i]



first_sum = cosine_table_self(mem_sum)
np.fill_diagonal(first_sum,0)

img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(first_sum, sorted(mydict.keys()), sorted(mydict.keys()), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)


second_sum = cosine_table_self(first_sum)
np.fill_diagonal(second_sum,0)
img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(second_sum, sorted(mydict.keys()), sorted(mydict.keys()), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)


##############3
# expected first order
import generalized_jaccard as jac
import pandas as pd

first = jac.get_jaccard_matrix(wd,2,second_order=False)
np.fill_diagonal(first,0)
first = pd.DataFrame(first)
first.columns = mydict.keys()
first.index = mydict.keys()
#first.to_csv('distributions/points/first_order_20.csv',index=False)

img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(first, list(first.columns), list(first.columns), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)
#plt.show()
#fig.savefig('heatmaps/glove.png', dpi=100, transparent = True)

first = jac.get_jaccard_matrix(wd,2,second_order=True)
np.fill_diagonal(first,0)
first = pd.DataFrame(first)
first.columns = mydict.keys()
first.index = mydict.keys()
#first.to_csv('distributions/points/first_order_20.csv',index=False)

img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(first, list(first.columns), list(first.columns), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)