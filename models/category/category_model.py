from __future__ import print_function


import sys
import os
import json
from models.category.hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from models.category.train import Graph
from models.category.data_load import get_batch_data, load_vocab, load_data
from scipy.stats import spearmanr

class CategoryModel:
    def __init__(self):

        self.graph = Graph(mode="dev");
        print("Graph loaded")

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(hp.logdir)); print("Model restored!")

    def eval(self,test_data):
        '''
        Get a Spearman rank-order correlation coefficient.

        Args:
          mode: A string. Either `val` or `test`.
        '''

        text_lengths, texts, categories = load_data(mode="dev",data=test_data)
        hp.batch_size=len(texts)

        # Parse
        X = np.zeros((len(texts), hp.max_len), np.int32)
        for i, text in enumerate(texts):
            text = np.fromstring(text, np.int32)
            X[i, -len(text):] = text

        # Feed-forward
        ys, preds = [], []
        for step in range(len(X) // hp.batch_size):
            # batch
            x = X[step * hp.batch_size: (step + 1) * hp.batch_size]
            y = categories[step * hp.batch_size: (step + 1) * hp.batch_size]

            ys.extend(y)


            # predict
            logits, gs = self.sess.run([self.graph.logits, self.graph.global_step], {self.graph.x: x, self.graph.y:y}) # (N, len(cat))
            predictions=logits.tolist()

            final_predictions=[]
            for i in range(len(predictions)):
                scores=[]
                for j in range(len(predictions[i])):
                  scores.append({"score":predictions[i][j],"category":hp.categories[j]})
                sorted_scores = sorted(scores, key=lambda k: k['score'],reverse=True)
                final_predictions.append(sorted_scores[0])


        return final_predictions

