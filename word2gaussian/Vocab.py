# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:33:46 2016

@author: whr94621
"""
from gzip import GzipFile
from collections import defaultdict
import numpy.random as random
import numpy

class Vocabulary(object):
    def __init__(self, id2w, w2id, size):
        
        self._word2id = w2id
        self._id2word = id2w
        self._len = size
    
    def __len__(self):
        return self._len
    
    def word2id(self, string):
        return self._word2id[string]
    
    def random_ids(self, N, UNW = True):
        random.seed(N)
        return random.randint(self._len)
    
    def tokenize(self, text, UNW = True):
        ids = []
        for word in text.strip().split(' '):
            ids.append(self.word2id(word))
        return numpy.array(ids)
    
    @classmethod    
    def load(cls, vocab_file, seq='\t'):
        id2w=[]        
        w2id=defaultdict(int)
        id2w.append('UNW')
        w2id['UNW'] = 0
        with GzipFile(vocab_file, 'r') as fin:
            for idx, line in enumerate(fin):
                word, freq = line.strip().split(seq)
                id2w.append(word)
                w2id[word] = idx
            
            size = idx + 2
        return cls(id2w, w2id, size)

if __name__ == '__main__':
    import os
    work_dir = os.getcwd()
    with open(work_dir + '/wiki_sample', 'r') as fin:
        text = fin.readline()
    vocab = Vocabulary.load(work_dir+'/vocab.corpus.gz')