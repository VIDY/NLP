from __future__ import print_function


import os
import json
from models.emoji.hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from models.emoji.train import Graph
from models.emoji.data_load import get_batch_data, load_vocab, load_data
from scipy.stats import spearmanr
import sys
import json
from tqdm import tqdm

class EmojiModel:
    def __init__(self):


      self.graph = Graph(mode="dev");
      print("Graph loaded")

      self.sess = tf.Session()
      self.saver = tf.train.Saver()
      self.saver.restore(self.sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")


    def eval(self,test_data):

        '''
        Get evaluation accuray.
        '''

        # Load data
        texts, Y = load_data(mode="dev",data=test_data)

        hp.batch_size=len(texts)
        print(hp.task_num)
        print(hp.batch_size)
        print(texts)
        print(Y)

        # Parse
        X = np.zeros((len(texts), hp.max_len), np.int32)
        for i, text in enumerate(texts):
            text = np.fromstring(text, np.int32)
            X[i, -len(text):] = text

        if hp.task_num == 2:
            Y = [np.fromstring(label, np.int32) for label in Y]

    
        # Restore parameters
        #saver.restore(self.sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

        with open('{}/eval.txt'.format(hp.logdir), 'a') as fout:
            # Feed-forward
            preds, seqlens = [], []
            for step in tqdm(range(len(X) // hp.batch_size)):
                # batch
                x = X[step * hp.batch_size: (step + 1) * hp.batch_size]
                # y = labels[step * hp.batch_size: (step + 1) * hp.batch_size]
                #
                # ys.extend(y)

                # predict
                if hp.task_num == 1:
                    preds_, gs = self.sess.run([self.graph.preds, self.graph.global_step], {self.graph.x: x}) # (N,)
                elif hp.task_num == 2:
                    # preds_, = self.sess.run([self.graph.preds], {self.graph.x: x})  # (N, K)
                    # print(preds_)
                    preds_, seqlens_, gs = self.sess.run([self.graph.preds, self.graph.seqlens, self.graph.global_step], {self.graph.x: x})  # (N, K)
                    seqlens.extend(seqlens_.tolist())
                preds.extend(preds_.tolist())

            # calculation
            if hp.task_num == 1:
                num0, num1, correct0, correct1 = 0, 0, 0, 0
                for y, pred in zip(Y, preds):
                    if y==0:
                        num0 += 1
                        if y == pred: correct0 += 1
                    else:
                        num1 += 1
                        if y == pred: correct1 += 1

                acc0 = correct0 / float(num0)
                acc1 = correct1 / float(num1)
                acc = (correct0+correct1) / float(num0+num1)

                fout.write('gs: %05d, acc0: %d/%d=%.02f, acc1: %d/%d=%.02f, acc: %d/%d=%.02f\n'
                           %(gs, correct0, num0, acc0, correct1, num1, acc1,
                             correct0+correct1, num0+num1, acc))
            elif hp.task_num == 2:


                print("pred:" + str(preds))
                #with open('/var/www/html/nlp/output/emoji.txt', 'w') as fout:
                #  json.dump(preds,fout)

                hits, predictions, labels = 0, 0, 0

                print(seqlens)

                final_results=[]
                for y, pred, seqlen in zip(Y, preds, seqlens):
                    # y: ?, pred: K, seqlen: scalar
                    seqlen = max(1, seqlen)


                    pred = pred[:seqlen] # -> pred <= K
                    print(y)
                    print(pred)
                    print(seqlen)

                    # pred = [0, 1, 2]
                    labeled_pred=[]
                    for unlabeled_pred in pred:
                      labeled_pred.append(hp.labels[unlabeled_pred])

                    final_results.append(labeled_pred)

                    hits += len(np.intersect1d(y, pred))
                    predictions += len(pred)
                    labels += len(y)

                return final_results
                with open('/var/www/html/nlp/output/emoji.txt', 'w') as fout:
                  json.dump(final_results,fout)


                #precision = hits / float(predictions + 0.0000001)
                #recall = hits / float(labels + 0.0000001)
                #f1 = 2. * precision * recall / (precision+recall)

                #fout.write("-------------\n")
                #fout.write('gs: %d\n' % gs)
                #fout.write('precision: %d/%d=%.02f\n' % (hits, predictions, precision))
                #fout.write('recall: %d/%d=%.02f\n' % (hits, labels, recall))
                #fout.write('f1 score: %.02f\n' % f1)
