# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 17:02:33 2016

@author: whr94621
"""

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)

from gzip import GzipFile

from word2gauss import GaussianEmbedding, iter_pairs
from vocab import Vocabulary

import sys

vocab = Vocabulary.load(sys.argv[2])

embed = GaussianEmbedding(len(vocab),50,covariance_type='diagonal',energy_type='KL',mu_max = 4.0, sigma_min = 1,
        sigma_max = 2)

with GzipFile(sys.argv[1], 'r') as corpus:
    for i in xrange(50):
        embed.train(iter_pairs(corpus, vocab, iterations=20), n_workers=16)

embed.save('embedding.tar.gz', vocab=vocab.id2word, full=True)
