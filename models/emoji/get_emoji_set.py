#-*- coding: utf-8 -*-
import codecs
import ast
from collections import Counter

num_samples = 1000000 # To save time, we look at the first 1M entries.
i = 0
emojis = []
for line in codecs.open('tweets/train.txt', 'r', 'utf8'):
    i += 1
    if i > num_samples: break

    try:
        d = ast.literal_eval(line.strip())
        emojis.extend(d["emojis"])
    except:
        continue

with codecs.open('tweets/emojis.txt', 'w', 'utf8') as fout:
    c = Counter(emojis)
    for emoji, cnt in c.most_common(len(c)):
        fout.write("%s\t%0.4f\n" %(emoji, cnt/float(len(emojis))))



