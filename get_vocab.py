# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:34:57 2016

@author: whr94621
"""

from pyspark import SparkConf, SparkContext
import sys
import io
import re

filter_en = re.compile('[^a-zA-Z-_]')



conf = SparkConf().setMaster("local").setAppName("MakeVocab")
sc = SparkContext(conf = conf)

corpus = sc.textFile(sys.argv[1])
s_word = sys.argv[2]

corpus = corpus.flatMap(lambda x: x.split(" "))

corpus = corpus.filter(lambda x: not filter_en.search(x))

corpus = corpus.map(lambda x: (x,1))

stop_word = sc.textFile(s_word)
stop_word = stop_word.map(lambda x: (x,1))
corpus = corpus.subtractByKey(stop_word)

corpus = corpus.reduceByKey(lambda x,y: x+y)

corpus = corpus.filter(lambda x: x[1]>100)

corpus = corpus.map(lambda x: x[0]+u'\t'+str(x[1]))

corpus = corpus.repartition(1)

corpus.saveAsTextFile('result')

