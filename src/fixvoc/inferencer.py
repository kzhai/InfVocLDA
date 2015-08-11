import sys, re, time, string
import numpy;
import scipy;
import scipy.special;
import nltk;

def compute_dirichlet_expectation(dirichlet_parameter):
    if (len(dirichlet_parameter.shape) == 1):
        return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter))
    return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter, 1))[:, numpy.newaxis]

class Inferencer:
    def __init__(self,
                 hash_oov_words=False,
                 compute_elbo=True
                 ):
        self._hash_oov_words = hash_oov_words;
        self._compute_elbo = compute_elbo;
        
    def _initialize(self,
                    vocab,
                    number_of_documents,
                    number_of_topics,
                    alpha_theta,
                    alpha_eta,
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
        self._alpha_theta = alpha_theta
        self._alpha_eta = alpha_eta
        self._tau = tau0 + 1
        self._kappa = kappa
        self._counter = 0
        
        self._epsilon = pow(self._tau + self._counter, -self._kappa)

        # Initialize the variational distribution q(beta|lambda)
        self._beta = 1*numpy.random.gamma(100., 1./100., (self._number_of_topics, self._vocab_size))
        self._exp_E_log_beta = numpy.exp(compute_dirichlet_expectation(self._beta));

    def parse_doc_list(self, docs):
        raise NotImplementedError;
    
    def e_step(self, wordids):
        raise NotImplementedError;

    def m_step(self, batch_size, sstats):
        # rhot will be between 0 and 1, and says how much to weight the information we got from this mini-batch.
        self._epsilon = pow(self._tau + self._counter, -self._kappa)
        
        # update lambda based on documents.
        self._beta = self._beta * (1-self._epsilon) + self._epsilon * (self._alpha_eta + self._number_of_documents * sstats / batch_size);
        expect_log_beta = compute_dirichlet_expectation(self._beta);
        self._exp_E_log_beta = numpy.exp(expect_log_beta);
        
        corpus_level_elbo = 0;
        
        if self._compute_elbo:
            corpus_level_elbo += numpy.sum((self._alpha_eta - self._beta) * expect_log_beta);
            corpus_level_elbo += numpy.sum(scipy.special.gammaln(self._beta) - scipy.special.gammaln(self._alpha_eta));
            corpus_level_elbo += numpy.sum(scipy.special.gammaln(self._alpha_eta * self._vocab_size) - scipy.special.gammaln(numpy.sum(self._beta, 1)))
        
        return corpus_level_elbo;

    def learning(self, docs):
        raise NotImplementedError;

    """
    """
    def export_beta(self, exp_beta_path, top_display=-1):
        self._exp_E_log_beta = numpy.exp(compute_dirichlet_expectation(self._beta));
        
        output = open(exp_beta_path, 'w');
        for k in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" %(k));
            freqdist = nltk.probability.FreqDist();
            freqdist.clear();

            for word in self._vocab.keys():
                #freqdist.inc(word, self._exp_E_log_beta[k, self._vocab[word]]);
                freqdist[word]+=self._exp_E_log_beta[k, self._vocab[word]];
                
            i=0;
            for key in freqdist.keys():
                i += 1;
                output.write(key + "\t" + str(freqdist[key]) + "\n");
                if top_display>0 and i>=top_display:
                    break;
