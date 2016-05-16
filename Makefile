build:
	#build corpus 
	~/anaconda2/bin/python generate_corpus.py  ./dataset/wiki.corpus.gz wiki.new 0.5
	gzip wiki.new 
vocab: wiki.new.gz
	#make vocabulary from corpus
	~/spark-1.6.0-bin-hadoop2.6/bin/spark-submit get_vocab.py wiki.new.gz ./dataset/stop_words.txt
	mv ./result/part-* vocab
	rm -rf ./result
filter: vocab
	#filter the word that have synsets in Wordnet
	~/anaconda2/bin/python filter.py vocab
	rm vocab
	mv vocab.new vocab
	gzip vocab

