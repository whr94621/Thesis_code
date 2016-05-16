'''
This script use to test the result of word2gauss

'''
from tarfile import open as open_tar
from contextlib import closing
import json
import numpy
import scipy.stats as stats
'''
fname = sys.argv[2]
vocab_f = sys.argv[1]
query = sys.argv[3]
metric = sys.argv[4]
'''


def logKL(mu1,sigma1,mu2,sigma2):
    k = mu1.shape[0]
    _mu = mu2 - mu1
    result = numpy.sum(sigma1/sigma2)+numpy.sum(_mu/sigma2*_mu)-k+numpy.log(numpy.prod(sigma2)/numpy.prod(sigma1))
    return numpy.log(result/2.0)


def logIP(mu1,sigma1,mu2,sigma2):
    d = mu1.shape[0]
    _mu = mu1-mu2
    _sigma = sigma1+sigma2

    return numpy.log((2 * numpy.pi)**(-d/2.0) * numpy.prod(_sigma)**(-1/2.0) * numpy.exp(- numpy.sum(_mu * _mu / _sigma) / 2.0))

def cosine(v1,v2):
    return numpy.dot(v1,v2)/numpy.dot(v1,v1)**0.05/numpy.dot(v2,v2)**0.5

def Mahalanobis(mu_1, sigma_1, mu_2, sigma_2):
    _mu = mu_1 - mu_2
    _sigma = sigma_1 + sigma_2

    return numpy.sum(_mu / _sigma * _mu)


class Embedding:
    def __init__(self, embedding_file, vocab):
        self._vocab = vocab
        with open_tar(embedding_file, 'r') as fin:
            with closing(fin.extractfile('parameters')) as f:
                params = json.loads(f.read())
                self.N = params['N']
                self.K = params['K']

            self.mu = numpy.empty([self.N, self.K], dtype=numpy.float)
            self.context = numpy.empty([self.N, self.K], dtype=numpy.float)
            self.sigma = numpy.empty([2 * self.N, self.K], dtype=numpy.float)

            with closing(fin.extractfile('word_mu')) as f:
                for i, line in enumerate(f):
                    vec = line.strip().split()[1:]
                    self.mu[i,:] = [float(ele) for ele in vec]
            with closing(fin.extractfile('mu_context')) as f:
                for i, line in enumerate(f):
                    vec = line.strip().split()
                    self.context[i,:] = [float(ele) for ele in vec]
            with closing(fin.extractfile('sigma')) as f:
                for i, line in enumerate(f):
                    vec = line.strip().split()
                    self.sigma[i,:] = [float(ele) for ele in vec]

    def nearest_word(self, word, flag='IP', number=10):
        distance = numpy.zeros(self.N, dtype=numpy.float)
        word_id = self._vocab.word2id(word)
        '''
        if flag == 'IP':
            #word_IP = IP(self.mu[word_id,:], self.sigma[word_id,:], self.mu[word_id,:], self.sigma[word_id,:])
            for i in xrange(self.N):
                distance[i] = IP(self.mu[word_id,:], self.sigma[word_id,:], self.mu[i,:], self.sigma[i,:]) / \
                    IP(self.mu[i,:],self.sigma[i,:],self.mu[i,:],self.sigma[i,:])
        '''
        if flag == 'Mahalanobis':
            for i in xrange(self.N):
                distance[i] = Mahalanobis(self.mu[word_id,:], self.sigma[word_id,:],
                                        self.mu[i,:], self.sigma[word_id,:])

        elif flag == 'cosine':
            for i in xrange(self.N):
                distance[i] = cosine(self.mu[word_id,:],self.mu[i,:])

        elif flag == 'IP':
            for i in xrange(self.N):
                distance[i] = logIP(self.mu[i,:], self.sigma[i,:], self.mu[word_id,:], self.sigma[word_id,:]) - logIP(self.mu[i,:], self.sigma[i,:], self.mu[i,:], self.sigma[i,:])

        candid = numpy.argsort(-distance)[1:number]
        result = [(self._vocab.id2word(ele),distance[ele]) for ele in candid]
        return result

    def get_embedding(self, word):
        idx = self._vocab.word2id(word)
        return [word, (self.mu[idx,:],self.sigma[idx,:])]

    def get_variance_determint(self,word):
        idx = self._vocab.word2id(word)
        return numpy.prod(self.sigma[idx,:])

    def test_similarity(self, test_file, isCombined=True,
                        hasTitle=True, flag='cosine'):
        score_human = []
        score_model = []
        with open(test_file,'r') as fin:
            if hasTitle:
                fin.readline()
            for line in fin:
                lline = line.strip().split('\t')
                id_1 = self._vocab.word2id(lline[0].lower())
                id_2 = self._vocab.word2id(lline[1].lower())
                score_human.append(float(lline[2]))

                if flag == 'cosine':
                    s = cosine(self.mu[id_1,:], self.mu[id_2,:])
                elif flag == 'IP':
                    s = logIP(self.mu[id_1,:],self.sigma[id_1,:],
                              self.mu[id_2,:],self.sigma[id_2,:])

                score_model.append(s)
        coeff = stats.pearsonr(numpy.array(score_human),numpy.array(score_model))
        print coeff


if __name__ == '__main__':
    from vocab import Vocabulary
    import os
    work_dir = os.getcwd()
    test_file = work_dir + '/dataset/wordsim353/combined'
    vocab = Vocabulary.load(work_dir + '/dataset/vocab.new.gz')
    embed = Embedding(work_dir + '/embedding_result/Result/May5/embedding.tar.gz',
                      vocab)


