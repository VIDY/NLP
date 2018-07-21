from __future__ import print_function
import numpy as np
import os
import sys
import json
from operator import itemgetter

class RelatedModel:
    def __init__(self):

        self.vocabulary={}
        for i, line in enumerate(open("data/glove.6B/glove.6B.300d.txt", 'r')):
            entry=line.split()
            word=entry[0]
            entry.pop(0)
            wordvec=entry
            for j in range(len(wordvec)):
                wordvec[j]=float(wordvec[j])
            self.vocabulary[word]=wordvec

        print("Done loading")

    def euclidean_dist(self,vec1, vec2):
        vec1=np.asarray(vec1)
        vec2=np.asarray(vec2)
        return np.sqrt(np.sum((vec1-vec2)**2))

    def search(self,word):
        word_vector1=self.vocabulary[word]

        results=[]
        for word in self.vocabulary:
            word_vector2=self.vocabulary[word]
            distance=self.euclidean_dist(word_vector1,word_vector2)
            if distance>0:
                results.append({"word":word,"distance":distance})
        results_sorted = sorted(results, key=itemgetter('distance'), reverse=False)
        return results_sorted[:10]

    def eval(self,test_data):

        return self.search(test_data["word"])
