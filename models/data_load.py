from __future__ import print_function
from models.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import re, regex
import codecs
import unicodedata
from tqdm import tqdm
import json

def load_vocab(num_vocab=hp.num_vocab):
    vocab = ["<PAD>", "<UNK>", "<EOS>"]
    for i, line in enumerate(open(hp.glove, 'r')):
        if i == num_vocab - 3: break
        word = line.split()[0]
        vocab.append(word)

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word

def load_labels():
    cat2idx = {cat: idx for idx, cat in enumerate(hp.categories)}
    idx2cat = {idx: cat for idx, cat in enumerate(hp.categories)}
    return cat2idx, idx2cat

def load_data(mode="train",data=[]):
    word2idx, idx2word = load_vocab()
    cat2idx, idx2cat = load_labels()

    if mode == "train":
        f = hp.train
    elif mode == "dev":
        f = hp.dev
    else:
        f = hp.test

    #test_data=[{"text": "CLEVELAND \\u2014 You suddenly don\\u2019t hear much around the N.B.A. about the majesty of LeBron James\\u2019s eight consecutive trips to the N.B.A. finals.", "category": "theater"}, {"text": "The justified awe inspired by James\\u2019s feat \\u2014 his entree into a club whose only prior members played for the Bill Russell-era Boston Celtics in the 1950s and \\u201960s \\u2014 didn\\u2019t even last two weeks.", "category": "theater"}, {"text": "Blame Kevin Durant for that. For the second straight June, Durant coldly strolled into Quicken Loans Arena for a Game 3 in the finals and was just too much, too ruthless, for James and his Cleveland Cavaliers.", "category": "theater"}, {"text": "Only this time, Durant didn\\u2019t merely usher James to the brink of a humbling sweep on the game\\u2019s biggest stage. With his 43-point, 13-rebound, 7-assist masterpiece on Wednesday night, Durant started the clock on James\\u2019s third foray into free agency.", "category": "theater"}, {"text": "Fretting has thus quickly replaced fawning as the reflex reaction in this part of the world when it comes to the subject of the Akron, Ohio-reared James. This playoff series (and the season) technically isn\\u2019t over. But it realistically is \\u2014 and the locals know it.", "category": "theater"}, {"text": "During this season\\u2019s playoffs, James has accounted for nearly a third of his team\\u2019s box-score statistics.", "category": "theater"}, {"text": "They are surely aware that N.B.A. teams are 0-131 in N.B.A. history when facing a 3-0 series deficit. They likewise understand that the odds of James electing to stay with his home-state Cavaliers this off-season appear to be only marginally better than the prospect of four consecutive wins against the mighty Warriors \\u2014 unless merely getting to the championship round is enough for James at 33.", "category": "theater"}, {"text": "Those privy to James\\u2019s thinking say that at this stage, pleading from family members appears to be the only force that could persuade him to extend his second stint with the Cavs and resist the opportunity to switch teams, as he did in 2010 and again in 2014.", "category": "theater"}, {"text": "The leaguewide belief, of course, is that chasing championships is James\\u2019s priority, which necessitates relocating to a team far better equipped to do so than the Cavaliers. He can do so either by signing elsewhere as a free agent after July 1 or opting into the final season of his current Cleveland contract and forcing a trade to a new home.", "category": "theater"}, {"text": "The prominent ESPN commentator Stephen A. Smith reported Wednesday that James plans to speak to six external suitors in addition to the Cavaliers. Philadelphia, Houston, Miami and the Los Angeles Lakers were all mentioned \\u2014 as were the juggernaut that has inflicted seven losses in James\\u2019s last eight finals games (Golden State) and his longtime Eastern Conference nemesis (Boston).", "category": "theater"}, {"text": "If Paul stays with the Rockets, Houston becomes the closest thing to a favorite on my scorecard, no matter how complicated it would be for the 65-win Rockets and their general manager, Daryl Morey, to orchestrate the requisite salary-cap gymnastics to bring James in. But don\\u2019t discount the idea that James could try to bring Paul with him to a team that can afford two superstars, such as the Lakers, because he and Paul really are that close.", "category": "theater"}, {"text": "I tend to believe the suggestions in circulation that even James himself isn\\u2019t sure yet where he\\u2019s headed. Yet I also suspect he has much more of an inkling about his summer plans than he would ever consent to let on while his Cavaliers are still playing. He\\u2019s too studious, too into preparation, not to have some thoughts stored up.", "category": "theater"}, {"text": "The challenge is finding the best situation out there that puts James on something more closely resembling a level playing field with the Warriors and their four All-Stars.", "category": "theater"}, {"text": "James will have three full weeks to dig into the various options, study the landscape for the most fertile locales for title contention and, perhaps most crucially, make clandestine connections with the starry likes of Paul George, Kawhi Leonard and, yes, Paul and James Harden to explore the prospects of teaming up somewhere.", "category": "theater"}, {"text": "It frankly wouldn\\u2019t surprise me to ultimately learn that James contacts Durant \\u2014 himself a free-agent-to-be if he declines his player option with the Warriors for next season, as expected \\u2014 to see if joining forces somewhere makes any sense. James, remember, is the face of the N.B.A.\\u2019s Player Power era. Anyone who tries to tell him that it simply wouldn\\u2019t be fair for James and Durant to wear the same uniform should prepare for James\\u2019s retort that he has zero interest in the outside world\\u2019s rules.", "category": "theater"}, {"text": "But this is as far as I\\u2019ll go with guarantees before the LeBron Sweepstakes officially begin: James will not feel as though he owes Cleveland anything after these last four seasons.", "category": "theater"}, {"text": "Not after the Cavaliers responded to the Warriors\\u2019 addition of Durant by chasing off their accomplished general manager David Griffin and giving in to the All-Star guard Kyrie Irving\\u2019s trade demand in August.", "category": "theater"}, {"text": "Nor after everything James said in his unforgettable \\u201cI\\u2019m Coming Home\\u201d essay he co-wrote with Lee Jenkins for Sports Illustrated in July 2014.", "category": "theater"}, {"text": "\\u201cMy goal is still to win as many titles as possible, no question,\\u201d James wrote. \\u201cBut what\\u2019s most important for me is bringing one trophy back to Northeast Ohio.\\u201d", "category": "theater"}, {"text": "One is all he managed, true, and that number will be recorded as an unmitigated disappointment on some scorecards out there.", "category": "theater"}, {"text": "So LeBron can leave now, knowing that he lifted the Cleveland Curse in 2016 by teaming with Irving to lead the Cavaliers to the first resurrection from a 3-1 finals deficit in league history. It was Cleveland\\u2019s first major sports championship since 1964 \\u2014 to go with all the other ways James financially revitalized this city after returning from the Miami Heat to rejoin the team that drafted him.", "category": "theater"}, {"text": "You will hear much leading into Friday night\\u2019s Game 4 about James\\u2019s unflattering finals record, which is poised to drop to 3-6. You will be reminded that only Jerry West (eight times) and Elgin Baylor (seven) have tasted finals disappointment more often.", "category": "theater"}, {"text": "Just don\\u2019t forget that it\\u2019s James\\u2019s unwavering brilliance, above all, that convinced the Warriors \\u2014 after a 73-win season in 2015-16 \\u2014 that they had to do anything they could to add Durant to Stephen Curry, Klay Thompson, Draymond Green and Andre Iguodala to ensure they could beat this one guy.", "category": "theater"}, {"text": "\\u201cThey go 73-9,\\u201d James lamented, \\u201cand then you add one of the best players that the N.B.A. has ever seen.\\u201d", "category": "theater"}, {"text": "The league doesn\\u2019t hand out rings for that, but maybe it should.", "category": "theater"}, {"text": "An earlier version of this column misstated the year of the city of Cleveland\\u2019s last major sports championship before the Cavaliers\\u2019 N.B.A. title in 2016. It was 1964 \\u2014 when the Browns defeated the Baltimore Colts in the N.F.L. championship game \\u2014 not 1954.", "category": "theater"}]
    # Parse
    text_lengths, texts, categories = [], [], []

    for entry in data:
      print(entry)
      text, category = entry["text"],"theater"

      text = text.lower()
      text = ''.join(char for char in unicodedata.normalize('NFD', text)
                     if unicodedata.category(char) != 'Mn')  # Strip accents
      text = regex.sub("[^ A-Za-z0-9\-']", "", text)
      text = [word2idx.get(word, 1) for word in text.split() + ["<EOS>"]]
      text = text[-hp.max_len:]
      text_lengths.append(len(text))
      texts.append(np.array(text, np.int32).tostring())

      category = cat2idx[category]
      categories.append(category)

    # Monitor

    print("text lengths look like", text_lengths[:10])
    print("texts look like", " ".join(idx2word[t] for t in np.fromstring(texts[0], np.int32)))
    print("categories look like", categories[:10])
    #print("test_data", json.dumps(test_data))

    return text_lengths, texts, categories

def get_batch_data():
    with tf.device('/cpu:0'):
        # Load data
        text_lengths, texts, categories = load_data()

        # calc total batch count
        num_batch = len(text_lengths) // hp.batch_size

        # Create Queues
        text_length, text, category = tf.train.slice_input_producer([text_lengths, texts, categories])

        # str to int
        text = tf.decode_raw(text, tf.int32)
        text = tf.pad(text, [(hp.max_len, 0)])[-hp.max_len:] # prepadding

        # Batching
        texts, categories = tf.train.batch([text, category],
                                           num_threads=8,
                                           shapes=([hp.max_len,], []),
                                           batch_size=hp.batch_size,
                                           capacity=hp.batch_size * 4,
                                           allow_smaller_final_batch=False)

        # _, (texts, categories) = tf.contrib.training.bucket_by_sequence_length(
        #                                                 input_length=text_length,
        #                                                 tensors=[text, category],
        #                                                 batch_size=hp.batch_size,
        #                                                 bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 10)],
        #                                                 num_threads=8,
        #                                                 capacity=hp.batch_size * 4,
        #                                                 dynamic_pad=True)

    return texts, categories, num_batch  # (N, T), (N,), ()

