#-*- coding: utf-8 -*-
'''
Splits three datasets, that is, sexual data, tweets for emoji binary prediction,
and tweets for hashtag prediction
into train, dev, and test.
Note that it is assumed three original raw files are placed in
`porno/xvideos.com-db.csv`,
`sad/tweets.txt`, and
`hashtag/*.json`.
'''
import codecs
from random import shuffle
import ast
from glob import glob

#1. porno
lines = codecs.open("porno/xvideos.com-db.csv", "r", "utf8").read().splitlines()
shuffle(lines)
num = len(lines)
train, dev, test = lines[:8 * num // 10], lines[8 * num // 10: 9 * num // 10], lines[9 * num // 10:]

with codecs.open('porno/train.txt', 'w', 'utf8') as fout: fout.write("\n".join(train))
with codecs.open('porno/dev.txt', 'w', 'utf8') as fout: fout.write("\n".join(dev))
with codecs.open('porno/test.txt', 'w', 'utf8') as fout: fout.write("\n".join(test))

#2. tweets (sad vs. not_sad)
i = 0
sad, not_sad = [], []
for line in codecs.open("sad/tweets.txt", "r", "utf8"):
    i += 1
    if i % 10000 == 0: print("num=", i, "# sad=", len(sad))

    try:
        d = ast.literal_eval(line.strip())
    except: # Igore some in incorrect format!
        continue

    text = d["description"]
    emojis = d["emojis"]

    if len(text) > 0 and len(emojis) > 0:
        if "Crying face" in emojis or\
           "Crying cat face" in emojis or\
           "Loudly crying face" in emojis:
            sad.append(text)
        elif len(not_sad) < len(sad):
            not_sad.append(text)

num_samples = min(len(sad), len(not_sad))
sad, not_sad = sad[:num_samples], not_sad[:num_samples]
shuffle(sad)
shuffle(not_sad)
train_s, dev_s, test_s = sad[:8 * num_samples // 10], sad[8 * num_samples // 10: 9 * num_samples // 10], sad[9 * num_samples // 10:]
train_n, dev_n, test_n = not_sad[:8 * num_samples // 10], not_sad[8 * num_samples // 10: 9 * num_samples // 10], not_sad[9 * num_samples // 10:]

with codecs.open('sad/train_s.txt', 'w', 'utf8') as fout: fout.write("\n".join(train_s))
with codecs.open('sad/dev_s.txt', 'w', 'utf8') as fout: fout.write("\n".join(dev_s))
with codecs.open('sad/test_s.txt', 'w', 'utf8') as fout: fout.write("\n".join(test_s))

with codecs.open('sad/train_n.txt', 'w', 'utf8') as fout: fout.write("\n".join(train_n))
with codecs.open('sad/dev_n.txt', 'w', 'utf8') as fout: fout.write("\n".join(dev_n))
with codecs.open('sad/test_n.txt', 'w', 'utf8') as fout: fout.write("\n".join(test_n))

# 3. tweets (hashtags)
i = 0
sents = []
files = glob('hashtag/*.json')
for f in files:
    print(f)
    for line in codecs.open(f, "r", "utf8"):
        i += 1
        if i % 10 != 0: continue # Every 10th element is taken.

        if "#" not in line: continue
        sents.append(line.strip())

shuffle(sents)
num_samples = len(sents)
train, dev, test= sents[:int(num_samples*.8)], sents[int(num_samples*.8):int(num_samples*.9)], sents[int(num_samples*.9):]
with codecs.open('hashtag/train.txt', 'w', 'utf8') as fout: fout.write("\n".join(train))
with codecs.open('hashtag/dev.txt', 'w', 'utf8') as fout: fout.write("\n".join(dev))
with codecs.open('hashtag/test.txt', 'w', 'utf8') as fout: fout.write("\n".join(test))


