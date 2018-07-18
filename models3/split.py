#-*- coding: utf-8 -*-
'''
Split two datasets, that is, sexual dataset (porno) and tweets,
into train, dev, and test.
Note that I assume they are placed in porno/xvideos.com-db.csv and
tweets/tweets.txt, respectively.
'''
import codecs, os
from random import shuffle

# porno
lines = codecs.open("porno/xvideos.com-db.csv", "r", "utf8").read().splitlines()
shuffle(lines)
num = len(lines)
train, dev, test = lines[:8 * num // 10], lines[8 * num // 10: 9 * num // 10], lines[9 * num // 10:]

with codecs.open('porno/train.txt', 'w', 'utf8') as fout:
    fout.write("\n".join(train))
with codecs.open('porno/dev.txt', 'w', 'utf8') as fout:
    fout.write("\n".join(dev))
with codecs.open('porno/test.txt', 'w', 'utf8') as fout:
    fout.write("\n".join(test))

# tweet
i=0
lines = []
for line in codecs.open("tweets/tweets.txt", "r", "utf8"):
    lines.append(line.strip())
    i+=1
    if i%1000000==0:
        shuffle(lines)
        num = len(lines)
        train, dev, test = lines[:8 * num // 10], lines[8 * num // 10: 9 * num // 10], lines[9 * num // 10:]

        with codecs.open('tweets/train.txt', 'a', 'utf8') as fout:
            fout.write("\n".join(train))
        with codecs.open('tweets/dev.txt', 'a', 'utf8') as fout:
            fout.write("\n".join(dev))
        with codecs.open('tweets/test.txt', 'a', 'utf8') as fout:
            fout.write("\n".join(test))
        lines = []


# lines = codecs.open("tweets/tweets.txt", "r", "utf8").read().splitlines()
# shuffle(lines)
# num = len(lines)
# train, dev, test = lines[:8 * num // 10], lines[8 * num // 10: 9 * num // 10], lines[9 * num // 10:]
#
# with codecs.open('tweets/train.txt', 'w', 'utf8') as fout:
#     fout.write("\n".join(train))
# with codecs.open('tweets/dev.txt', 'w', 'utf8') as fout:
#     fout.write("\n".join(dev))
# with codecs.open('tweets/test.txt', 'w', 'utf8') as fout:
#     fout.write("\n".join(test))


