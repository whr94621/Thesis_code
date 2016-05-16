from nltk.corpus import wordnet as wn
import sys

vocab_file = sys.argv[1]

with open(vocab_file, 'r') as fin, open('vocab.new', 'w') as fout:
    for line in fin:
        freq = int(line.strip().split('\t')[1])
        if freq >= 500:
            fout.write(line)

                
