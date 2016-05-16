# -*- coding: utf-8 -*-
"""
Created on Fri May 13 23:37:01 2016

@author: whr94621
"""
from word2gaussian import word2gauss, iter_pairs
import sys
import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)
from Vocab import Vocabulary
import gzip


vocab = Vocabulary.load(sys.argv[2])
embed = word2gauss(N=len(vocab), size=50, covariance_type='diagonal',
                       energy_type='IP', mu_max=4.0, sigma_min=0.5, sigma_max=1)

with gzip.open(sys.argv[1], 'r') as corpus:
    embed.train(iter_pairs(corpus, vocab, batch_size=10, iterations=1),
                n_workers=20)

