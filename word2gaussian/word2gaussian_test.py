# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:58:32 2016

@author: whr94621
"""

import tensorflow as tf
import numpy as np

sess = tf.Session()



a = tf.placeholder(dtype=tf.int32, shape=[None, 2])

v = [[1,2,3],
     [4,5,6],
     [7,8,9],
     [10,11,12]
     ]
b = tf.Variable(v)

def ops()