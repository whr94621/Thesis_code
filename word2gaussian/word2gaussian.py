# -*- coding: utf-8 -*-
"""
Created on Mon May  9 22:34:06 2016

@author: whr94621
"""
import tensorflow as tf
import numpy as np
from itertools import islice
from tarfile import open as open_tar
import json
import logging
import numpy.random as random

def test_to_pairs(iters, N, D):
    for i in xrange(iters):
        job = random.randint(low=1, high=N, size=(D,3))
        yield 1, job

def text_to_pairs(text, random_gen, half_window_size=2, nsamples_per_word=1,
                  vocab_len=1000):
    npairs = sum(
        [2 * len(doc) * half_window_size * nsamples_per_word for doc in text])
    next_pair = 0
    pairs = np.empty(shape=(npairs, 3), dtype=np.uint32)
    for doc in text:
        doc_len = doc.shape[0]
        for i in xrange(doc_len):
            if doc[i] == 0:
                continue
            for j in xrange(i + 1, min(i + half_window_size + 1, doc_len)):
                if doc[j] == 0:
                    continue
                for k in range(nsamples_per_word):
                    pairs[next_pair, 0] = doc[i]
                    pairs[next_pair, 1] = doc[j] + vocab_len
                    pairs[next_pair, 2] = doc[k] + vocab_len
                    next_pair += 1
    return np.ascontiguousarray(pairs[:next_pair, :])


def iter_pairs(fin, vocab, batch_size=1, nsamples=2, window=5, iterations=10):
    for idx in xrange(iterations):
        batch = list(islice(fin, 10))
        while len(batch) > 0:
            text = [vocab.tokenize(doc) for doc in batch]
            pairs = text_to_pairs(text, random_gen=vocab.random_ids,
                                  half_window_size=window,
                                  nsamples_per_word=nsamples,
                                  vocab_len=len(vocab))
            yield idx, pairs
            batch = list(islice(fin, 10))
        fin.seek(0)

def energy_ip(mu_1, sigma_1, mu_2, sigma_2, d):
    _mu = mu_1 + mu_2
    _sigma = sigma_1 + sigma_2
    return -1.0/2 * tf.log(tf.reduce_prod(_sigma, 1)) \
    - 1.0/2 * (_mu * _mu / _sigma) - d/2.0 * tf.log(2 * np.pi)


def gauss_init(shape, mean, std):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=std)
    return tf.Variable(initial)


def const_init(shape, value):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)



class word2gauss:
    def __init__(self, N, size=100, covariance_type='spherical',
                 mu_max=2.0, sigma_min=0.7, sigma_max=1.5,
                 energy_type='IP', init_params={
                     'mu0': 0.5,
                     'sigma_mean0': 0.5,
                     'sigma_std0': 1.5},
                     step=0.1, Closs=0.1, mu=None, sigma=None):

        graph = tf.Graph()
        with graph.as_default():
            word_mu = const_init(shape=[N * 2, size], value=init_params['mu0'])

            if covariance_type == 'spherical':
                word_sigma = gauss_init(shape=[N * 2, 1],
                                    mean=init_params['sigma_mean0'],
                                    std=init_params['sigma_std0'])
            elif covariance_type == 'diagonal':
                word_sigma = gauss_init(shape=[N * 2, size],
                                    mean=init_params['sigma_mean0'],
                                    std=init_params['sigma_std0'])
            else:
                raise ValueError

            sess = tf.Session()


        self.N = N
        self.size = size
        self.Closs = Closs
        self.word_mu = word_mu
        self.word_sigma= word_sigma
        self.sess = sess
        self.graph = graph
        self.mu_max = mu_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.covariance_type = covariance_type
        self.energy_type = energy_type
        self.step = step


    def save(self, fname, vocab=None, full=True):
        from tempfile import NamedTemporaryFile

        def save_embed(a, name, fout):
            with NamedTemporaryFile() as tmp:
                np.savetxt(tmp, a ,fmt='%s')
                tmp.seek(0)
                fout.add(tmp.name, arcname=name)

        _w_mu = self.sess(self.word_mu)
        _w_sigma = self.sess(self.word_sigma)


        with open_tar(fname, 'w:gz') as fout:

            save_embed(_w_mu, 'word_mu', fout)
            save_embed(_w_sigma, 'word_sigma', fout)


            params = {
                'N': self.N,
                'K': self.size,
                'covariance_type': self.covariance_type,
                'energy_type': self.energy_type,
                'sigma_min': self.sigma_min,
                'sigma_max': self.sigma_min,
                'mu_max': self.mu_max,
                'step': self.step,
                'Closs': self.Closs}

            with NamedTemporaryFile() as tmp:
                tmp.write(json.dumps(params))
                tmp.seek(0)
                fout.add(tmp.name, arcname='parameters')


    def _build_graph(self, input_ids):
        w_ids = tf.slice(input_=input_ids, begin=[0,0], size=[-1, 1])
        c_ids = tf.slice(input_=input_ids, begin=[0,1], size=[-1, 1])
        neg_ids = tf.slice(input_=input_ids, begin=[0,2], size=[-1, 1])
        w_ids = tf.reshape(w_ids, [-1])
        c_ids = tf.reshape(c_ids, [-1])
        neg_ids = tf.reshape(neg_ids, [-1])
        w_mu = tf.nn.embedding_lookup(self.word_mu, w_ids)
        w_sigma =tf.nn.embedding_lookup(self.word_sigma, w_ids)
        c_mu = tf.nn.embedding_lookup(self.word_mu, c_ids)
        c_sigma = tf.nn.embedding_lookup(self.word_sigma, c_ids)
        neg_mu = tf.nn.embedding_lookup(self.word_mu, neg_ids)
        neg_sigma = tf.nn.embedding_lookup(self.word_mu, neg_ids)

        loss = tf.reduce_mean(tf.nn.relu(self.Closs -
                                    energy_ip(w_mu, w_sigma, c_mu, c_sigma, self.size)+
                                    energy_ip(w_mu, w_sigma, neg_mu, neg_sigma, self.size))
                                    )

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
                                                                    loss=loss)
        return optimizer


    def hard_regularizer_mu(self, regularize_ids):
            op_mu_b = tf.scatter_update(self.word_mu, regularize_ids,
                                 tf.maximum(tf.nn.embedding_lookup(
                                 self.word_mu, regularize_ids), -self.mu_max))
            op_mu_t = tf.scatter_update(self.word_mu, regularize_ids,
                                 tf.minimum(tf.nn.embedding_lookup(
                                 self.word_mu, regularize_ids), self.mu_max))
            return [op_mu_b, op_mu_t]

    def hard_regularizer_sigma(self, regularize_ids):
            op_sigma_b = tf.scatter_update(self.word_sigma, regularize_ids,
                                 tf.maximum(tf.nn.embedding_lookup(
                                 self.word_sigma, regularize_ids), self.sigma_min))
            op_sigma_t = tf.scatter_update(self.word_sigma, regularize_ids,
                                 tf.minimum(tf.nn.embedding_lookup(
                                 self.word_sigma, regularize_ids), self.sigma_max))
            return [op_sigma_b, op_sigma_t]



    def train(self, iter_pairs, n_workers=4, report_interval=100):
        from Queue import Queue
        from threading import Thread, Lock
        import time

        with self.graph.as_default():
            self.sess.run(tf.initialize_all_variables())
            input_ids = tf.placeholder(shape=[None, 3], dtype=tf.int32)
            regularize_ids = tf.placeholder(shape=[None], dtype=tf.int32)
            optimizer = self._build_graph(input_ids=input_ids)
            regulizer = []
            regulizer += self.hard_regularizer_mu(regularize_ids) +\
                self.hard_regularizer_sigma(regularize_ids)
            jobs = Queue(maxsize=2 * n_workers)
            lock = Lock()
            processed = [0, report_interval, report_interval]
            t0 = time.time()

            def worker():
                while True:
                    iters, job = jobs.get()
                    if job is None:
                        break

                    self.sess.run(optimizer, feed_dict={input_ids:job})
                    self.sess.run(regulizer,
                                  feed_dict={regularize_ids:job.flatten()})

                    with lock:
                        processed[0] += 1
                        if processed[1] and processed[0] >= processed[1]:
                            t2 = time.time()
                            logging.info("Processed %d jobs in %d th interation, elapsed time: %s"
                             % (processed[0], iters + 1, t2 - t0))
                            processed[1] = processed[0] + processed[2]

            jobs = Queue(maxsize= 2 * n_workers)
            threads = []
            for k in xrange(n_workers):
                thread = Thread(target=worker)
                thread.daemon = True
                thread.start()
                threads.append(thread)

            for job in iter_pairs:
                jobs.put(job)
            for k in xrange(len(threads)):
                jobs.put((None, None))

            for thread in threads:
                thread.join()




'''
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)
    from Vocab import Vocabulary
    from gzip import GzipFile
    work_dir = '/home/whr94621/Documents/Thesis/code/dataset/'

    vocab = Vocabulary.load(work_dir + 'vocab.corpus.gz')
    embed = word2gauss(N=len(vocab), size=50, covariance_type='diagonal',
                       energy_type='IP', mu_max=4.0, sigma_min=0.5, sigma_max=1)

    with GzipFile(work_dir + 'wiki.corpus.gz', 'r') as corpus:
        embed.train(iter_pairs(corpus, vocab, iterations=1), n_workers=20)

    embed.train(test_to_pairs(100000, 5000, 50), n_workers=4)
'''