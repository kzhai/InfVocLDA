import sys, re, time, string
import numpy;
import scipy;
import scipy.special;
import nltk;

from inferencer import compute_dirichlet_expectation;
from inferencer import Inferencer;

class Hybrid(Inferencer):
    def __init__(self,
                 hash_oov_words=False,
                 number_of_samples=10,
                 burn_in_sweeps=5
                 ):
        Inferencer.__init__(self, hash_oov_words);
        
        self._number_of_samples = number_of_samples;
        self._burn_in_sweeps = burn_in_sweeps;

    def parse_doc_list(self, docs):
        if (type(docs).__name__ == 'str'):
            temp = list()
            temp.append(docs)
            docs = temp
    
        D = len(docs)
        
        wordids = list()
        for d in range(0, D):
            docs[d] = docs[d].lower()
            docs[d] = re.sub(r'-', ' ', docs[d])
            docs[d] = re.sub(r'[^a-z ]', '', docs[d])
            docs[d] = re.sub(r' +', ' ', docs[d])
            words = string.split(docs[d])
            ddict=[];
            for word in words:
                if (word in self._vocab):
                    ddict.append(self._vocab[word])
                else:
                    if self._hash_oov_words:
                        ddict.append(hash(word) % len(self._vocab));
            if len(ddict)==0:
                print >> sys.stderr, 'warning: document collapsed during parsing...' 
            wordids.append(ddict);
    
        return wordids;
    
    def e_step(self, wordids):
        batchD = len(wordids)
        
        document_level_elbo = 0;

        sufficient_statistics = numpy.zeros((self._number_of_topics, self._vocab_size));

        # Initialize the variational distribution q(theta|gamma) for the mini-batch
        batch_document_topic_distribution = numpy.zeros((batchD, self._number_of_topics));

        # Now, for each document d update that document's gamma and phi
        for d in xrange(batchD):
            phi = numpy.random.random((self._number_of_topics, len(wordids[d])));
            phi = phi / numpy.sum(phi, axis=0)[numpy.newaxis, :];
            phi_sum = numpy.sum(phi, axis=1)[:, numpy.newaxis];
            assert(phi_sum.shape == (self._number_of_topics, 1));

            for it in xrange(self._number_of_samples):
                for n in xrange(len(wordids[d])):
                    id = wordids[d][n];
                    
                    phi_sum -= phi[:, n][:, numpy.newaxis];
                    
                    # this is to get rid of the underflow error from the above summation, ideally, phi will become all integers after few iterations
                    phi_sum *= phi_sum > 0;
                    #assert(numpy.all(phi_sum >= 0));

                    temp_phi = (phi_sum + self._alpha_theta).T * self._exp_E_log_beta[:, wordids[d][n]];
                    assert(temp_phi.shape == (1, self._number_of_topics));
                    temp_phi /= numpy.sum(temp_phi);

                    # sample a topic for this word
                    temp_phi = numpy.random.multinomial(1, temp_phi[0, :])[:, numpy.newaxis];
                    assert(temp_phi.shape == (self._number_of_topics, 1));
                    
                    phi[:, n][:, numpy.newaxis] = temp_phi;
                    phi_sum += temp_phi;

                    # discard the first few burn-in sweeps
                    if it < self._burn_in_sweeps:
                        continue;
                    
                    sufficient_statistics[:, id] += temp_phi[:, 0];
                    
            batch_document_topic_distribution[d, :] = self._alpha_theta + phi_sum.T[0, :];
            
            if self._compute_elbo:
                document_level_elbo += len(wordids[d]);

                gammad = batch_document_topic_distribution[d];
                document_level_elbo += numpy.sum((self._alpha_theta - gammad) * numpy.exp(compute_dirichlet_expectation(gammad)));
                document_level_elbo += numpy.sum(scipy.special.gammaln(gammad) - scipy.special.gammaln(self._alpha_theta));
                document_level_elbo += numpy.sum(scipy.special.gammaln(self._alpha_theta * self._number_of_topics) - scipy.special.gammaln(numpy.sum(gammad)));

        sufficient_statistics /= (self._number_of_samples - self._burn_in_sweeps);
        
        if self._compute_elbo:
            document_level_elbo *= self._number_of_documents / batchD;

        return batch_document_topic_distribution, sufficient_statistics, document_level_elbo

    def learning(self, docs):
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp
        wordids = self.parse_doc_list(docs)
        (gamma, sstats, document_level_elbo) = self.e_step(wordids);
        corpus_level_elbo = self.m_step(len(docs), sstats);
        self._counter += 1;
        
        return gamma, document_level_elbo + corpus_level_elbo