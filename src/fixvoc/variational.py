"""
Online Variational Bayesian Inference for Latent Dirichlet Allocation

This code was modified from the code originally written by Matthew Hoffman.
Implements online VB for LDA as described in (Hoffman et al. 2010).

@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import sys, re, time, string
import numpy
import scipy
import scipy.special
import nltk;

from inferencer import compute_dirichlet_expectation;
from inferencer import Inferencer;

class Variational(Inferencer):
    def __init__(self,
                 hash_oov_words=False,
                 maximum_gamma_update_iteration=50,
                 minimum_mean_change_threshold=1e-3
                 ):
        Inferencer.__init__(self, hash_oov_words);
        
        self._maximum_gamma_update_iteration = maximum_gamma_update_iteration;
        self._minimum_mean_change_threshold = minimum_mean_change_threshold;

    def parse_doc_list(self, docs):
        if (type(docs).__name__ == 'str'):
            temp = list()
            temp.append(docs)
            docs = temp
    
        D = len(docs)
        
        wordids = list()
        wordcts = list()
        for d in range(0, D):
            docs[d] = docs[d].lower()
            docs[d] = re.sub(r'-', ' ', docs[d])
            docs[d] = re.sub(r'[^a-z ]', '', docs[d])
            docs[d] = re.sub(r' +', ' ', docs[d])
            words = string.split(docs[d])
            ddict = dict()
            for word in words:
                if (word in self._vocab):
                    wordtoken = self._vocab[word]
                    if (not wordtoken in ddict):
                        ddict[wordtoken] = 0
                    ddict[wordtoken] += 1
                else:
                    if self._hash_oov_words:
                        wordtoken = hash(word) % len(self._vocab);
                        if (not wordtoken in ddict):
                            ddict[wordtoken] = 0
                        ddict[wordtoken] += 1
            wordids.append(ddict.keys())
            wordcts.append(ddict.values())
    
        return wordids, wordcts

    def e_step(self, wordids, wordcts):
        batch_size = len(wordids)

        document_level_elbo = 0;

        # Initialize the variational distribution q(theta|gamma) for the mini-batch
        gamma = 1*numpy.random.gamma(100., 1./100., (batch_size, self._number_of_topics))
        exp_E_log_theta = numpy.exp(compute_dirichlet_expectation(gamma))

        sstats = numpy.zeros(self._beta.shape)
        # Now, for each document d update that document's gamma and phi
        meanchange = 0
        for d in range(0, batch_size):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            exp_E_log_theta_d = exp_E_log_theta[d, :]
            exp_E_log_beta_d = self._exp_E_log_beta[:, ids]
            # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w. phi_norm is the normalizer.
            phi_norm = numpy.dot(exp_E_log_theta_d, exp_E_log_beta_d) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, self._maximum_gamma_update_iteration):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time. Substituting the value of the optimal phi back into the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha_theta + exp_E_log_theta_d * numpy.dot(cts / phi_norm, exp_E_log_beta_d.T)
                exp_E_log_theta_d = numpy.exp(compute_dirichlet_expectation(gammad))
                phi_norm = numpy.dot(exp_E_log_theta_d, exp_E_log_beta_d) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = numpy.mean(abs(gammad - lastgamma))
                if (meanchange < self._minimum_mean_change_threshold):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient statistics for the M step.
            sstats[:, ids] += numpy.outer(exp_E_log_theta_d.T, cts/phi_norm)
                        
            if self._compute_elbo:
                document_level_elbo += numpy.sum(cts * phi_norm)

                # E[log p(theta | alpha) - log q(theta | gamma)]
                document_level_elbo += numpy.sum((self._alpha_theta - gammad) * exp_E_log_theta_d);
                document_level_elbo += numpy.sum(scipy.special.gammaln(gammad) - scipy.special.gammaln(self._alpha_theta));
                document_level_elbo += numpy.sum(scipy.special.gammaln(self._alpha_theta * self._number_of_topics) - scipy.special.gammaln(numpy.sum(gammad)));

        # This step finishes computing the sufficient statistics for the M step, so that sstats[k, w] = \sum_d n_{dw} * phi_{dwk} = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._exp_E_log_beta

        if self._compute_elbo:
            document_level_elbo *= self._number_of_documents / batch_size;

        return gamma, sstats, document_level_elbo

    def learning(self, docs):
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp
        (wordids, wordcts) = self.parse_doc_list(docs)
        (gamma, sstats, document_level_elbo) = self.e_step(wordids, wordcts);
        corpus_level_elbo = self.m_step(len(docs), sstats);
        self._counter += 1;
        
        return gamma, document_level_elbo + corpus_level_elbo

    def approx_bound(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        (wordids, wordcts) = self.parse_doc_list(docs)
        batch_size = len(docs)

        score = 0
        Elogtheta = compute_dirichlet_expectation(gamma)
        expElogtheta = numpy.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batch_size):
            gammad = gamma[d, :]
            ids = wordids[d]
            cts = numpy.array(wordcts[d])
            phinorm = numpy.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = numpy.log(sum(numpy.exp(temp - tmax))) + tmax
            score += numpy.sum(cts * phinorm)
#             oldphinorm = phinorm
#             phinorm = n.dot(expElogtheta[d, :], self._exp_E_log_beta[:, ids])
#             print oldphinorm
#             print n.log(phinorm)
#             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += numpy.sum((self._alpha_theta - gamma)*Elogtheta)
        score += numpy.sum(gammaln(gamma) - gammaln(self._alpha_theta))
        score += sum(gammaln(self._alpha_theta*self._number_of_topics) - gammaln(numpy.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._number_of_documents / len(docs)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + numpy.sum((self._alpha_eta-self._beta)*self._Elogbeta)
        score = score + numpy.sum(gammaln(self._beta) - gammaln(self._alpha_eta))
        score = score + numpy.sum(gammaln(self._alpha_eta*self._vocab_size) - 
                              gammaln(numpy.sum(self._beta, 1)))

        return(score)