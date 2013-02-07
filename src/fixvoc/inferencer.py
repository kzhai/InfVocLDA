import sys, re, time, string
import numpy;
import scipy;
import scipy.special;
import nltk;

'''
def compute_dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha)))
    return(scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha, 1))[:, numpy.newaxis])
'''

def compute_dirichlet_expectation(dirichlet_parameter):
    if (len(dirichlet_parameter.shape) == 1):
        return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter))
    return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter, 1))[:, numpy.newaxis]

class Inferencer:
    def __init__(self,
                 hash_oov_words=False,
                 compute_elbo=True
                 ):
        numpy.random.seed(100000001);
        
        self._hash_oov_words = hash_oov_words;
        self._compute_elbo = compute_elbo;
        
    def _initialize(self,
                    vocab,
                    number_of_topics,
                    number_of_documents,
                    alpha,
                    eta,
                    tau0,
                    kappa
                    ):
        self._vocab = dict()
        for word in vocab:
            word = word.lower()
            word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)

        self._number_of_topics = number_of_topics
        self._vocab_size = len(self._vocab)
        self._number_of_documents = number_of_documents
        self._alpha = alpha
        self._eta = eta
        self._tau = tau0 + 1
        self._kappa = kappa
        self._counter = 0
        
        self._epsilon = pow(self._tau + self._counter, -self._kappa)

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*numpy.random.gamma(100., 1./100., (self._number_of_topics, self._vocab_size))
        self._exp_expect_log_beta = numpy.exp(compute_dirichlet_expectation(self._lambda));

    def parse_doc_list(self, docs):
        raise NotImplementedError;
    
    '''    
    def e_step(self, wordids):
        raise NotImplementedError;
        
        batchD = len(wordids)

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

                    temp_phi = (phi_sum + self._alpha).T * self._exp_expect_log_beta[:, wordids[d][n]];
                    assert(temp_phi.shape == (1, self._number_of_topics));
                    temp_phi /= numpy.sum(temp_phi);

                    # sample a topic for this word
                    temp_phi = numpy.random.multinomial(1, temp_phi[0])[:, numpy.newaxis];
                    assert(temp_phi.shape == (self._number_of_topics, 1));
                    
                    phi[:, n][:, numpy.newaxis] = temp_phi;
                    phi_sum += temp_phi;

                    # discard the first few burn-in sweeps
                    if it < self._burn_in_sweeps:
                        continue;
                    
                    sufficient_statistics[:, id] += temp_phi[:, 0];

            batch_document_topic_distribution[d, :] = self._alpha + phi_sum.T[0, :];

        sufficient_statistics /= (self._number_of_samples - self._burn_in_sweeps);

        return (batch_document_topic_distribution, sufficient_statistics)
    '''

    def m_step(self, batch_size, sstats):
        # rhot will be between 0 and 1, and says how much to weight the information we got from this mini-batch.
        self._epsilon = pow(self._tau + self._counter, -self._kappa)
        
        # update lambda based on documents.
        self._lambda = self._lambda * (1-self._epsilon) + self._epsilon * (self._eta + self._number_of_documents * sstats / batch_size);
        expect_log_beta = compute_dirichlet_expectation(self._lambda);
        self._exp_expect_log_beta = numpy.exp(expect_log_beta);
        
        corpus_level_elbo = 0;
        
        if self._compute_elbo:
            corpus_level_elbo += numpy.sum((self._eta - self._lambda) * expect_log_beta);
            corpus_level_elbo += numpy.sum(scipy.special.gammaln(self._lambda) - scipy.special.gammaln(self._eta));
            corpus_level_elbo += numpy.sum(scipy.special.gammaln(self._eta * self._vocab_size) - scipy.special.gammaln(numpy.sum(self._lambda, 1)))
        
        return corpus_level_elbo;

    def learning(self, docs):
        raise NotImplementedError;

    """
    """
    def export_beta(self, exp_beta_path, top_display=-1):
        self._exp_expect_log_beta = numpy.exp(compute_dirichlet_expectation(self._lambda));
        
        output = open(exp_beta_path, 'w');
        for k in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" %(k));
            freqdist = nltk.probability.FreqDist();
            freqdist.clear();

            for word in self._vocab.keys():
                freqdist.inc(word, self._exp_expect_log_beta[k, self._vocab[word]]);
                
            i=0;
            for key in freqdist.keys():
                i += 1;
                output.write(key + "\t" + str(freqdist[key]) + "\n");
                if top_display>0 and i>=top_display:
                    break;