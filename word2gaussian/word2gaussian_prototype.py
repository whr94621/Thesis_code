# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:57:44 2016

@author: whr94621
"""

from Vocab import Vocabulary
import os
import tensorflow as tf
import numpy
import numpy.random as random
import logging
from Queue import Queue
from threading import Thread, Lock
import time

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)

work_dir = os.getcwd()

vocab= Vocabulary.load(work_dir + '/vocab.corpus.gz')

embed_size = len(vocab)
embed_dim =50
patch_size = 50
n_workers = 20
report_interval = 100
# word representation
word_mu = tf.Variable(tf.random_uniform([embed_size, embed_dim], -1.0, 1.0))
word_sigma = tf.Variable(tf.random_uniform([embed_size, embed_dim], 0.5, 1.0))
context_mu = tf.Variable(tf.random_uniform([embed_size, embed_dim], -1.0, 1.0))
context_sigma = tf.Variable(
    tf.random_uniform([embed_size, embed_dim], 0.5, 1.0))

# train_patch
'''
train_patch is a [patch_size, 3] 2-D tensor, which is like
[[w_0, c_0, neg_0],
 [w_1, c_1, neg_1],
 ....
 [w_n, c_n, neg_n]]
'''
train_patch = tf.placeholder(shape=[None, 3], dtype=tf.int32)

regulize_patch = tf.placeholder(shape=[None], dtype=tf.int32)

# build graph

w_ids = tf.slice(
    train_patch, begin=[0,0], size=[patch_size,1], name="word_ids")

c_ids = tf.slice(
    train_patch, begin=[0,1], size=[patch_size,1], name="context_ids")

neg_ids = tf.slice(
    train_patch, begin=[0,2], size=[patch_size,1], name="negative_samples_ids")

def hard_regularizer(margin):
    '''
    if isinstance(margin, float):
        op_b = tf.scatter_update(embed, input_ids,
                                 tf.maximum(tf.nn.embedding_lookup(
                                 embed, input_ids), -margin))
        op_t = tf.scatter_update(embed, input_ids,
                                 tf.minimum(tf.nn.embedding_lookup(
                                 embed, input_ids), margin))
    '''
    if True:




        op_b = tf.scatter_update(word_mu, regulize_patch, tf.maximum(tf.nn.embedding_lookup(word_mu, regulize_patch), margin[0]), use_locking=True)

        op_t = tf.scatter_update(word_mu, regulize_patch, tf.minimum(tf.nn.embedding_lookup(word_mu, regulize_patch), margin[1]), use_locking=True)

        return [op_b, op_t]
    else:
        raise ValueError

def energy_ip(mu_1, sigma_1, mu_2, sigma_2, d):

    _mu = mu_1 + mu_2
    _sigma = sigma_1 + sigma_2
    return -1.0/2 * tf.log(tf.reduce_prod(_sigma, 1)) \
    - 1.0/2 * (_mu * _mu / _sigma) - d/2.0 * tf.log(2 * numpy.pi)


w_mu = tf.nn.embedding_lookup(word_mu, w_ids)
w_sigma = tf.nn.embedding_lookup(word_sigma, w_ids)
c_mu = tf.nn.embedding_lookup(context_mu, c_ids)
c_sigma = tf.nn.embedding_lookup(context_sigma, c_ids)
neg_mu = tf.nn.embedding_lookup(context_mu, neg_ids)
neg_sigma = tf.nn.embedding_lookup(context_sigma, neg_ids)

loss = tf.reduce_mean(tf.nn.relu(5 - energy_ip(w_mu, w_sigma, c_mu, c_sigma, 50) \
       + energy_ip(w_mu, w_sigma, neg_mu, neg_sigma, 50)))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

op_b, op_t = hard_regularizer([-0.5, 0.5])
#training

sess = tf.Session()
sess.run(tf.initialize_all_variables())

jobs = Queue(maxsize = 2*n_workers)
lock = Lock()
processed = [0, report_interval, report_interval]
t1 = time.time()

def worker():
    while True:
        job = jobs.get()
        if job is None:
            break
        '''
        sess.run(optimizer, feed_dict={train_patch:job})
        '''
        job_w = job[:,0].flatten()
        sess.run([op_b, op_t], feed_dict={regulize_patch:job_w})
        with lock:
            processed[0] += 1
            if processed[1] and processed[0] >= processed[1]:
                t2 = time.time()
                logging.info("Processed %d jobs, elapsed time: %s" % (processed[0], t2-t1) )
                processed[1] = processed[0] + processed[2]
jobs = Queue(maxsize = 2*n_workers)


threads = []
for k in xrange(n_workers):
    thread = Thread(target=worker)
    thread.daemon = True
    thread.start()
    threads.append(thread)

for idx in xrange(10000):
    job = random.randint(low=0, high=embed_size, size=(50,3))
    jobs.put(job)

for idx in len(threads):
    jobs.put(None)

for thread in threads:
    thread.join()


