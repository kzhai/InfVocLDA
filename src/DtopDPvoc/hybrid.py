import sys, re, time, string, random, time, math;
import numpy;
import scipy;
import scipy.special;
import scipy.io;
import scipy.sparse;
import nchar;

import nltk;
import nltk.corpus;

numpy.random.seed(100000001)

"""
Implements online variational Bayesian for LDA.
"""
class Hybrid:
    """
    """
    def __init__(self,
                 minimum_word_length=3,
                 maximum_word_length=20,
                 dict_list=None,
                 N=3,
                 word_model_smooth=1e6,
                 char_list=string.lowercase
                 #char_list=string.lowercase + string.digits
                 ):
        from nltk.stem.porter import PorterStemmer
        self._stemmer = PorterStemmer();

        self._minimum_word_length = minimum_word_length;
        self._maximum_word_length = maximum_word_length;
        self._word_model_smooth = word_model_smooth;
        self._char_list = char_list;
        self._n_char_model = N;

        #'''
        if dict_list != None:
            tokens = [];
            for line in open(dict_list, 'r'):
                line = line.strip();
                if len(line) <= 0:
                    continue;
                #tokens.append(line);
                tokens.append(self._stemmer.stem(line));
            #tokens = set(tokens);
            self._word_model = nchar.NcharModel(self._n_char_model, tokens, self._word_model_smooth, self._maximum_word_length, self._minimum_word_length, self._char_list);

            '''
            print "successfully train the %d charactor word model." % (self._n_char_model);
            print "\tdictionary file %s" % (dict_list);
            print "\tword model smoothing %f" % (self._word_model_smooth);
            print "\tminimum word length %d" % (self._minimum_word_length);
            print "\tmaximum word length %d" % (self._maximum_word_length);
            print "\tvalid charactor list %s" % (self._char_list);
            '''
        else:
            '''
            from nltk.corpus import brown
            tokens = [];
            for word in brown.words():
                word = word.lower();
                word = re.sub(r'-', ' ', word);
                word = re.sub(r'[^a-z0-9 ]', '', word);
                word = re.sub(r' +', ' ', word);
                if len(word)==0:
                    continue;
                if word in nltk.corpus.stopwords.words('english'):
                    continue;
                
                tokens.append(stemmer.stem(word));
            self._word_model = nchar.NcharModel(self._n_char_model, tokens, self._word_model_smooth, self._maximum_word_length, self._minimum_word_length, self._char_list);
            print "successfully train the %d charactor word model." % (self._n_char_model);
            print "\tdictionary is brown corpus in nltk";
            print "\tword model smoothing %f" % (self._word_model_smooth);
            print "\tminimum word length %d" % (self._minimum_word_length);
            print "\tmaximum word length %d" % (self._maximum_word_length);
            print "\tvalid charactor list %s" % (self._char_list);
            '''

            self._word_model = None;
        #'''

        self._setting_title = "settings-";
        self._param_title = "param-";
        self._exp_beta_title = "exp_beta-";
        #self._new_word_title = "new_word-";
        
        self._gamma_title = "gamma-";
        
        self._nu_1_title = "nu_1-";
        self._nu_2_title = "nu_2-";
        self._index_title = "index-";
        self._nupos_title = "nupos-";
        self._ranking_statistics_title = "ranking-";
        self._trace_title = "trace-";
                
    """
    """
    def _initialize(self,
                    vocab,
                    number_of_topics,
                    number_of_documents,
                    batch_size,
                    expected_truncation_size,
                    alpha_theta=1e-2,
                    alpha_beta=1e6,
                    tau=1.,
                    kappa=0.5,
                    refine_vocab_interval=10,
                    save_word_trace=False,
                    ranking_smooth_factor=1e-12,
                    #gamma_converge_threshold=1e-3,
                    #number_of_samples=50,
                    number_of_samples=10,
                    #burn_in_sweeps=2
                    burn_in_sweeps=5
                    ):
        
        self._number_of_topics = number_of_topics;
        self._number_of_documents = number_of_documents;
        self._batch_size = batch_size;

        self._word_to_index = {};
        self._index_to_word = {};
        for word in set(vocab):
            self._index_to_word[len(self._index_to_word)] = word;
            self._word_to_index[word] = len(self._word_to_index);
        vocab = self._index_to_word.keys();
        
        self._new_words = [len(self._index_to_word)];
        
        self._word_trace=None;
        if save_word_trace:
            self._word_trace = [];
            for index in vocab:
                self._word_trace.append(numpy.zeros((self._number_of_topics, self._number_of_documents/self._batch_size + 1), dtype='int32') + numpy.iinfo(numpy.int32).max);
            
        self._index_to_nupos = [];
        self._nupos_to_index = [];
        for k in xrange(self._number_of_topics):
            self._index_to_nupos.append(dict());
            self._nupos_to_index.append(dict());
            
            random.shuffle(vocab);
            
            for index in vocab:
                self._nupos_to_index[k][len(self._nupos_to_index[k])] = index;
                self._index_to_nupos[k][index] = len(self._index_to_nupos[k]);
                
        self._truncation_size = [];
        self._truncation_size_prime = [];
        self._nu_1 = {};
        self._nu_2 = {};
        for k in xrange(self._number_of_topics):
            self._truncation_size.append(len(self._index_to_nupos[k]));
            self._truncation_size_prime.append(len(self._index_to_nupos[k]));
            self._nu_1[k] = numpy.ones((1, self._truncation_size[k]));
            self._nu_2[k] = numpy.ones((1, self._truncation_size[k]));

        self._expected_truncation_size = expected_truncation_size;
        
        self._alpha_theta = alpha_theta;
        self._alpha_beta = alpha_beta;
        self._tau = tau;
        self._kappa = kappa;
        
        #self._gamma_converge_threshold = gamma_converge_threshold;
        self._number_of_samples = number_of_samples;
        self._burn_in_sweeps = burn_in_sweeps;
        assert(self._burn_in_sweeps < self._number_of_samples);
        
        self._ranking_smooth_factor = ranking_smooth_factor;
        self._reorder_vocab_interval = refine_vocab_interval;
        
        self._ranking_statistics_scale = 1e30;
        
        self._counter = 0;

        self._ranking_statistics = [];
        for k in xrange(self._number_of_topics):
            self._ranking_statistics.append(nltk.probability.FreqDist());

            for index in self._index_to_nupos[k]:
                self._ranking_statistics[k].inc(index, self._ranking_smooth_factor);
                '''
                if self._word_model != None:
                    self._ranking_statistics[k].inc(index, self._word_model.probability(self._index_to_word[index]) * self._ranking_statistics_scale);
                else:
                    self._ranking_statistics[k].inc(index, self._ranking_smooth_factor);
                '''

        self._document_topic_distribution = None;
        
        if self._word_trace!=None:
            self.update_word_trace();

    def update_word_trace(self):
        if self._counter>self._number_of_documents/self._batch_size:
            return;
        for topic_index in xrange(self._number_of_topics):
            temp_keys = self._ranking_statistics[topic_index].keys();
            for word_rank in xrange(len(temp_keys)):
                self._word_trace[temp_keys[word_rank]][topic_index, self._counter:] = word_rank+1;
                
    def parse_doc_list(self, docs):
        if (type(docs).__name__ == 'str'):
            temp = list()
            temp.append(docs)
            docs = temp
    
        assert self._batch_size == len(docs);

        batch_documents = [];
        
        for d in xrange(self._batch_size):
            '''
            docs[d] = docs[d].lower();
            docs[d] = re.sub(r'-', ' ', docs[d]);
            docs[d] = re.sub(r'[^a-z ]', '', docs[d]);
            docs[d] = re.sub(r'[^a-z0-9 ]', '', docs[d]);
            docs[d] = re.sub(r' +', ' ', docs[d]);
            
            words = [];
            for word in docs[d].split():
                if word in nltk.corpus.stopwords.words('english'):
                    continue;
                word = self._stemmer.stem(word);
                if word in nltk.corpus.stopwords.words('english'):
                    continue;
                if len(word)>=self.maximum_word_length or len(word)<=self._minimum_word_length
                    continue;
                words.append(word);
            '''
            
            words = [word for word in docs[d].split() if len(word)<=self._maximum_word_length and len(word)>=self._minimum_word_length];
            
            document_topics = numpy.zeros((self._number_of_topics, len(words)));

            for word_index in xrange(len(words)):
                word = words[word_index];
                # valid only if limiting the ranking statistics 
                if word not in self._word_to_index:
                    #if this word never appeared before
                    index = len(self._word_to_index);

                    self._index_to_word[len(self._index_to_word)] = word;
                    self._word_to_index[word] = len(self._word_to_index);
                    
                    if self._word_trace!=None:                    
                        self._word_trace.append(numpy.zeros((self._number_of_topics, self._number_of_documents/self._batch_size + 1), dtype='int32') + numpy.iinfo(numpy.int32).max);
                        
                    for topic in xrange(self._number_of_topics):
                        self._ranking_statistics[topic].inc(index, self._ranking_smooth_factor);

                else:
                    index = self._word_to_index[word];
                        
                for topic in xrange(self._number_of_topics):
                    if index not in self._index_to_nupos[topic]:
                        # if this word is not in current vocabulary
                        self._nupos_to_index[topic][len(self._nupos_to_index[topic])] = index;
                        self._index_to_nupos[topic][index] = len(self._index_to_nupos[topic]);
                        
                        self._truncation_size_prime[topic] += 1;
                        
                    document_topics[topic, word_index]=self._index_to_nupos[topic][index];
                    
            batch_documents.append(document_topics);
            
        if self._word_trace!=None:
            self.update_word_trace();
        
        self._new_words.append(len(self._word_to_index));
        
        return batch_documents;

    """
    Compute the aggregate digamma values, for phi update.
    """
    def compute_exp_weights(self):
        exp_weights = {};
        exp_oov_weights = {};
        
        for k in xrange(self._number_of_topics):
            psi_nu_1_k = scipy.special.psi(self._nu_1[k]);
            psi_nu_2_k = scipy.special.psi(self._nu_2[k]);
            psi_nu_all_k = scipy.special.psi(self._nu_1[k] + self._nu_2[k]);
            
            aggregate_psi_nu_2_minus_psi_nu_all_k = numpy.cumsum(psi_nu_2_k - psi_nu_all_k, axis=1);
            exp_oov_weights[k] = numpy.exp(aggregate_psi_nu_2_minus_psi_nu_all_k[0, -1]);
            
            aggregate_psi_nu_2_minus_psi_nu_all_k = numpy.hstack((numpy.zeros((1, 1)), aggregate_psi_nu_2_minus_psi_nu_all_k[:, :-1]));
            assert(aggregate_psi_nu_2_minus_psi_nu_all_k.shape==psi_nu_1_k.shape);
            
            exp_weights[k] = numpy.exp(psi_nu_1_k - psi_nu_all_k + aggregate_psi_nu_2_minus_psi_nu_all_k);

        return exp_weights, exp_oov_weights;
    
    """
    """
    def e_step(self, batch_size, wordids, directory=None):
        sufficient_statistics = {};
        for k in xrange(self._number_of_topics):
            sufficient_statistics[k] = numpy.zeros((1, self._truncation_size_prime[k]));
            
        batch_document_topic_distribution = numpy.zeros((batch_size, self._number_of_topics));
        #batch_document_topic_distribution = scipy.sparse.dok_matrix((batch_size, self._number_of_topics), dtype='int16');

        #log_likelihood = 0;
        exp_weights, exp_oov_weights = self.compute_exp_weights();
        
        # Now, for each document d update that document's phi_d for every words
        for d in xrange(batch_size):
            phi = numpy.random.random(wordids[d].shape);
            phi = phi / numpy.sum(phi, axis=0)[numpy.newaxis, :];
            phi_sum = numpy.sum(phi, axis=1)[:, numpy.newaxis];
            #assert(phi_sum.shape == (self.number_of_topics, 1));

            for it in xrange(self._number_of_samples):
                for n in xrange(wordids[d].shape[1]):
                    phi_sum -= phi[:, n][:, numpy.newaxis];
                    # this is to get rid of the underflow error from the above summation, ideally, phi will become all integers after few iterations
                    phi_sum *= phi_sum > 0;
                    #assert(numpy.all(phi_sum >= 0));

                    temp_phi = phi_sum + self._alpha_theta;
                    #assert(temp_phi.shape == (self.number_of_topics, 1));
                    
                    for k in xrange(self._number_of_topics):
                        id = wordids[d][k, n];

                        if id >= self._truncation_size[k]:
                            # if this word is an out-of-vocabulary term
                            temp_phi[k, 0] *= exp_oov_weights[k];
                        else:
                            # if this word is inside current vocabulary
                            temp_phi[k, 0] *= exp_weights[k][0, id];

                    temp_phi /= numpy.sum(temp_phi);
                    #assert(temp_phi.shape == (self.number_of_topics, 1));

                    # sample a topic for this word
                    temp_phi = temp_phi.T[0];
                    temp_phi = numpy.random.multinomial(1, temp_phi)[:, numpy.newaxis];
                    #assert(temp_phi.shape == (self.number_of_topics, 1));
                    
                    phi[:, n][:, numpy.newaxis] = temp_phi;
                    phi_sum += temp_phi;
                    #assert(numpy.all(phi_sum >= 0));

                    # discard the first few burn-in sweeps
                    if it >= self._burn_in_sweeps:
                        for k in xrange(self._number_of_topics):
                            id = wordids[d][k, n];
                            sufficient_statistics[k][0, id] += temp_phi[k, 0];

            batch_document_topic_distribution[d, :] = self._alpha_theta + phi_sum.T[0, :];
                        
        for k in xrange(self._number_of_topics):
            sufficient_statistics[k] /= (self._number_of_samples - self._burn_in_sweeps);

        return sufficient_statistics, batch_document_topic_distribution;

    """
    """
    def m_step(self, batch_size, sufficient_statistics, close_form_updates=False):
        #sufficient_statistics = self.sort_sufficient_statistics(sufficient_statistics);
        reverse_cumulated_phi = {};
        for k in xrange(self._number_of_topics):
            reverse_cumulated_phi[k] = self.reverse_cumulative_sum_matrix_over_axis(sufficient_statistics[k], 1);
        
        if close_form_updates:
            self._nu_1 = 1 + sufficient_statistics;
            self._nu_2 = self._alpha_beta + reverse_cumulated_phi;
        else:
            # Epsilon will be between 0 and 1, and says how much to weight the information we got from this mini-batch.
            self._epsilon = pow(self._tau + self._counter, -self._kappa);
            
            self.update_accumulate_sufficient_statistics(sufficient_statistics);

            for k in xrange(self._number_of_topics):
                if self._truncation_size[k] < self._truncation_size_prime[k]:
                    self._nu_1[k] = numpy.append(self._nu_1[k], numpy.ones((1, self._truncation_size_prime[k] - self._truncation_size[k])), 1);
                    self._nu_2[k] = numpy.append(self._nu_2[k], numpy.ones((1, self._truncation_size_prime[k] - self._truncation_size[k])), 1);
                    
                    self._truncation_size[k] = self._truncation_size_prime[k];
                    
                self._nu_1[k] += self._epsilon * (self._number_of_documents / batch_size * sufficient_statistics[k] + 1 - self._nu_1[k]);
                self._nu_2[k] += self._epsilon * (self._alpha_beta + self._number_of_documents / batch_size * reverse_cumulated_phi[k] - self._nu_2[k]);

    """
    """
    def update_accumulate_sufficient_statistics(self, sufficient_statistics):
        for k in xrange(self._number_of_topics):
            for index in self._index_to_word:
                self._ranking_statistics[k].inc(index, -self._epsilon*self._ranking_statistics[k][index]);
            for index in self._index_to_nupos[k]:
                if self._word_model != None:
                    adjustment = self._word_model.probability(self._index_to_word[index]) * self._ranking_statistics_scale;
                else:
                    adjustment = 1.;
                self._ranking_statistics[k].inc(index, self._epsilon*adjustment*sufficient_statistics[k][0, self._index_to_nupos[k][index]]);

    """
    """
    def prune_vocabulary(self):
        # Re-order the nu values
        new_index_to_nupos = [];
        new_nupos_to_index = [];
        new_nu_1 = {};
        new_nu_2 = {};
        for k in xrange(self._number_of_topics):
            if len(self._index_to_nupos[k]) < self._expected_truncation_size:
                new_nu_1[k] = numpy.zeros((1, len(self._index_to_nupos[k])));
                new_nu_2[k] = numpy.zeros((1, len(self._index_to_nupos[k])));
            else:
                new_nu_1[k] = numpy.zeros((1, self._expected_truncation_size));
                new_nu_2[k] = numpy.zeros((1, self._expected_truncation_size));
            new_index_to_nupos.append(dict());
            new_nupos_to_index.append(dict());
            for index in self._ranking_statistics[k].keys():
                if len(new_index_to_nupos[k])>=min(self._index_to_nupos[k], self._expected_truncation_size):
                    break;
                
                #if index in words_to_keep and index in self._index_to_nupos[k].keys():
                new_nupos_to_index[k][len(new_index_to_nupos[k])] = index;
                new_index_to_nupos[k][index] = len(new_index_to_nupos[k]);
                
                # TODO: verify with jordan
                if index not in self._index_to_nupos[k]:
                    # TODO: this statement is never reached.
                    new_nu_1[k][0, new_index_to_nupos[k][index]] = 1;
                    new_nu_2[k][0, new_index_to_nupos[k][index]] = 1;
                else:
                    new_nu_1[k][0, new_index_to_nupos[k][index]] = self._nu_1[k][0, self._index_to_nupos[k][index]];
                    new_nu_2[k][0, new_index_to_nupos[k][index]] = self._nu_2[k][0, self._index_to_nupos[k][index]];

            self._truncation_size[k] = len(new_index_to_nupos[k]);
            self._truncation_size_prime[k] = self._truncation_size[k];
            
        self._index_to_nupos = new_index_to_nupos;
        self._nupos_to_index = new_nupos_to_index;
        self._nu_1 = new_nu_1;
        self._nu_2 = new_nu_2;

    """
    """
    def learning(self, batch):
        self._counter += 1;

        # This is to handle the case where someone just hands us a single document, not in a list.
        if (type(batch).__name__ == 'string'):
            temp = list();
            temp.append(batch);
            batch = temp;

        batch_size = len(batch);

        # Parse the document mini-batch
        clock = time.time();
        wordids = self.parse_doc_list(batch);
        clock_p_step = time.time() - clock;
        
        # E-step: hybrid approach, sample empirical topic assignment
        clock = time.time();
        sufficient_statistics, batch_document_topic_distribution = self.e_step(batch_size, wordids);
        clock_e_step = time.time() - clock;
        
        # M-step: online variational inference
        clock = time.time();
        self.m_step(batch_size, sufficient_statistics);
        if self._counter % self._reorder_vocab_interval==0:
            self.prune_vocabulary();
        clock_m_step = time.time() - clock;
        
        print 'P-step, E-step and M-step take %d, %d, %d seconds respectively...' % (clock_p_step, clock_e_step, clock_m_step);
        
        return batch_document_topic_distribution;

    """
    """
    def reverse_cumulative_sum_matrix_over_axis(self, matrix, axis):
        cumulative_sum = numpy.zeros(matrix.shape);
        (k, n) = matrix.shape;
        if axis == 1:
            for j in xrange(n - 2, -1, -1):
                cumulative_sum[:, j] = cumulative_sum[:, j + 1] + matrix[:, j + 1];
        elif axis == 0:
            for i in xrange(k - 2, -1, -1):
                cumulative_sum[i, :] = cumulative_sum[i + 1, :] + matrix[i + 1, :];
    
        return cumulative_sum;

    def export_beta(self, exp_beta_path, top_display=-1):
        exp_weights, exp_oov_weights = self.compute_exp_weights();

        output = open(exp_beta_path, 'w');
        for k in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (k));
            freqdist = nltk.probability.FreqDist();
            freqdist.clear();

            for index in self._index_to_nupos[k]:
                freqdist.inc(index, exp_weights[k][0, self._index_to_nupos[k][index]]);

            i = 0;
            for key in freqdist.keys():
                i += 1;
                output.write(self._index_to_word[key] + "\t" + str(freqdist[key]) + "\n");
                if top_display>0 and i>=top_display:
                    break
                
        output.close();

    def export_model_checkpoint(self, directory='../output/tmp/'):
        if not directory.endswith('/'):
            directory += "/";
            
        directory += self._setting_title;

        param_path = directory + self._param_title + str(self._counter);
        self.export_parameters(param_path);
        
        index_path = directory + self._index_title + str(self._counter);
        self.export_word_index(index_path);
        
        nupos_path = directory + self._nupos_title + str(self._counter);
        self.export_index_nupos(nupos_path);

        nu_1_path = directory + self._nu_1_title + str(self._counter);
        nu_2_path = directory + self._nu_2_title + str(self._counter);
        self.export_nu(nu_1_path, nu_2_path);

        ranking_path = directory + self._ranking_statistics_title + str(self._counter);
        self.export_ranking_statistics(ranking_path);
        
        if self._word_trace!=None:
            trace_path = directory + self._trace_title + str(self._counter);
            self.export_word_trace(trace_path);

    # TODO: add in counter
    def import_model_checkpoint(self, directory='../output/tmp/', counter=0):
        if not directory.endswith('/'):
            directory += "/";

        directory += self._setting_title;
        self.import_parameters(directory, counter);
        self.import_word_index(directory, counter);
        self.import_ranking_statistics(directory, counter);
        self.import_nu(directory, counter);
        self.import_index_nupos(directory, counter);
        
        self.import_word_trace(directory, counter);

    def export_parameters(self, settings_path):
        settings_output = open(settings_path, 'w');

        settings_output.write("alpha_theta=" + str(self._alpha_theta) + "\n");
        settings_output.write("alpha_beta=" + str(self._alpha_beta) + "\n");
        settings_output.write("tau0=" + str(self._tau) + "\n");
        settings_output.write("kappa=" + str(self._kappa) + "\n");
        
        settings_output.write("number_of_documents=" + str(self._number_of_documents) + "\n");
        settings_output.write("number_of_topics=" + str(self._number_of_topics) + "\n");
        settings_output.write("desired_truncation_level=" + str(self._expected_truncation_size) + "\n");

        settings_output.write("vocab_prune_interval=" + str(self._reorder_vocab_interval) + "\n");
        settings_output.write("batch_size=" + str(self._batch_size) + "\n");
        
        settings_output.write("number_of_samples=" + str(self._number_of_samples) + "\n");
        settings_output.write("burn_in_sweeps=" + str(self._burn_in_sweeps) + "\n");

        settings_output.write("ranking_smooth_factor=" + str(self._ranking_smooth_factor) + "\n");
        settings_output.write("ranking_statistics_scale=" + str(self._ranking_statistics_scale) + "\n");
        settings_output.write("counter=" + str(self._counter) + "\n");
        
        settings_output.write("truncation_level=");
        settings_output.write(" ".join([str(truncation_level) for truncation_level in self._truncation_size]) + "\n");

        settings_output.write("truncation_level_prime=");
        settings_output.write(" ".join([str(truncation_level) for truncation_level in self._truncation_size_prime]) + "\n");
        
        settings_output.close();

    def import_parameters(self, settings_path):
        settings_input = open(settings_path, 'r');
            
        self._alpha_theta = float(settings_input.readline().split('=')[1])
        self._alpha_beta = float(settings_input.readline().split('=')[1])
        self._tau = float(settings_input.readline().split('=')[1])
        self._kappa = float(settings_input.readline().split('=')[1])

        self._number_of_documents = int(settings_input.readline().split('=')[1])
        self._number_of_topics = int(settings_input.readline().split('=')[1])
        self._expected_truncation_size = int(settings_input.readline().split('=')[1])
        
        self._reorder_vocab_interval = int(settings_input.readline().split('=')[1]);
        self._batch_size = int(settings_input.readline().split('=')[1]);

        #self._gamma_converge_threshold = float(settings_input.readline().split('=')[1]);
        self._number_of_samples = int(settings_input.readline().split('=')[1]);
        self._burn_in_sweeps = int(settings_input.readline().split('=')[1]);
        
        self._ranking_smooth_factor = float(settings_input.readline().split('=')[1]);
        self._ranking_statistics_scale = float(settings_input.readline().split('=')[1]);
        self._counter = int(settings_input.readline().split('=')[1]);
        self._epsilon = pow(self._tau + self._counter, -self._kappa);

        self._truncation_size = [];
        #assert settings_input.readline().strip()=="truncation_level=";
        truncation = settings_input.readline().strip().split('=')[1];
        truncation = truncation.split();
        assert len(truncation)==self._number_of_topics
        for value in truncation:
            self._truncation_size.append(int(value));
            #self._truncation_size[k] = int(truncation[k]);
        assert len(self._truncation_size)==self._number_of_topics;    
                
        self._truncation_size_prime = [];
        #assert settings_input.readline().strip()=="truncation_level_prime=";
        truncation_prime = settings_input.readline().strip().split('=')[1];
        truncation_prime = truncation_prime.split();
        assert len(truncation_prime)==self._number_of_topics;
        for value in truncation_prime:
            self._truncation_size_prime.append(int(value));
            #self._truncation_size_prime[k] = int(truncation_prime[k]);
        assert len(self._truncation_size_prime)==self._number_of_topics;

    def export_word_index(self, word_index_path):
        settings_output = open(word_index_path, 'w');
        settings_output.write(" ".join([str(value) for value in self._new_words]) + "\n");
        for index in xrange(len(self._index_to_word)):
            settings_output.write("%s\n" % self._index_to_word[index]);
        settings_output.close();

    def import_word_index(self, word_index_path):
        settings_input = open(word_index_path, 'r');
        self._new_words = [int(value) for value in settings_input.readline().strip().split()];
        self._index_to_word = {};
        self._word_to_index = {};
        for line in settings_input:
            line = line.strip();
            self._index_to_word[len(self._index_to_word)] = line;
            self._word_to_index[line] = len(self._word_to_index);
            
    def export_index_nupos(self, index_nupos_path):
        settings_output = open(index_nupos_path, 'w');
        for k in xrange(self._number_of_topics):
            #settings_output.write(str(k) + "\t");
            for nupos in self._nupos_to_index[k]:
                settings_output.write(" %d=%d" % (nupos, self._nupos_to_index[k][nupos]));
            settings_output.write("\n");
        settings_output.close();

    def import_index_nupos(self, index_nupos_path):
        settings_input = open(index_nupos_path, 'r');
        self._index_to_nupos = [];
        self._nupos_to_index = [];
        for k in xrange(self._number_of_topics):
            self._index_to_nupos.append(dict());
            self._nupos_to_index.append(dict());
            nuposes = settings_input.readline().split();
            #assert nuposes[0] == str(k);
            for token in nuposes:
                tokens = token.split('=');
                self._nupos_to_index[k][int(tokens[0])] = int(tokens[1]);
                self._index_to_nupos[k][int(tokens[1])] = int(tokens[0]);

    def export_nu(self, nu_1_path, nu_2_path):
        settings_output = open(nu_1_path, 'w');
        for k in xrange(self._number_of_topics):
            #settings_output.write("nu_1 %d\n" % (k));
            for row in self._nu_1[k]:
                settings_output.write(" ".join([str(value) for value in row]) + "\n");
        settings_output.close();
                
        settings_output = open(nu_2_path, 'w');
        for k in xrange(self._number_of_topics):
            #settings_output.write("nu_2 %d\n" % (k));
            for row in self._nu_2[k]:
                settings_output.write(" ".join([str(value) for value in row]) + "\n");
        settings_output.close();

    def import_nu(self, nu_1_path, nu_2_path):
        settings_input = open(nu_1_path, 'r');
        self._nu_1 = {};
        for k in xrange(self._number_of_topics):
            nu_1_tokens = settings_input.readline().split();
            self._nu_1[k] = numpy.zeros((1, self._truncation_size[k]));
            count = 0;
            for value in nu_1_tokens:
                self._nu_1[k][0, count] = float(value);
                count += 1;

        settings_input = open(nu_2_path, 'r');
        self._nu_2 = {};
        for k in xrange(self._number_of_topics):
            nu_2_tokens = settings_input.readline().split();
            self._nu_2[k] = numpy.zeros((1, self._truncation_size[k]));
            count = 0;
            for value in nu_2_tokens:
                self._nu_2[k][0, count] = float(value);
                count += 1;

    def export_word_trace(self, word_trace_path):
        settings_output = open(word_trace_path, 'w');
        settings_output.write("%d\t%d\t%d\n" % (len(self._word_trace), self._number_of_topics, self._number_of_documents/self._batch_size + 1));
        for word_trace in self._word_trace:
            for row in word_trace:
                settings_output.write(" ".join([str(value) for value in row]) + "\n");
        settings_output.close();

    def import_word_trace(self, word_trace_path):
        settings_input = open(word_trace_path, 'r');
        self._word_trace = [];
        dimensions = settings_input.readline().strip().split();
        words_in_total = int(dimensions[0]);
        rows = int(dimensions[1]);
        cols = int(dimensions[2]);
        for index in xrange(words_in_total):
            #index = int(settings_input.readline().strip());
            #assert index == index;
            self._word_trace.append(numpy.zeros((rows, cols), dtype='int32'));
            for row_index in xrange(rows):
                count=0;
                for value in settings_input.readline().strip().split():
                    self._word_trace[index][row_index, count] = int(value);
                    count += 1;

    def export_ranking_statistics(self, ranking_statistics_path):
        settings_output = open(ranking_statistics_path, 'w');
        for k in xrange(self._number_of_topics):
            #settings_output.write(str(k) + "\t");
            for index in self._ranking_statistics[k].keys():
                settings_output.write(" %d=%f" % (index, self._ranking_statistics[k][index]));
            settings_output.write("\n");
        settings_output.close();

    def import_ranking_statistics(self, ranking_statistics_path):
        settings_input = open(ranking_statistics_path, 'r');
        self._ranking_statistics = {};
        for k in xrange(self._number_of_topics):
            self._ranking_statistics[k] = nltk.probability.FreqDist();
            ranking_statistics = settings_input.readline().split();
            #assert ranking_statistics[0]==str(k);
            for token in ranking_statistics:
                tokens = token.split('=');
                self._ranking_statistics[k].inc(int(tokens[0]), float(tokens[1]) + self._ranking_smooth_factor);

    """
    """
    def export_intermediate_gamma(self, directory='../output/tmp/'):
        if not directory.endswith('/'):
            directory += "/";
        
        if self._counter!=0:
            gamma_path = directory + self._gamma_title + str(self._counter) + ".txt";
            numpy.savetxt(gamma_path, self._document_topic_distribution);
            #scipy.io.mmwrite(gamma_path, self._document_topic_distribution);
            self._document_topic_distribution = None;
        
    """
    """
    def inference(self, batch):
        # This is to handle the case where someone just hands us a single document, not in a list.
        if (type(batch).__name__ == 'string'):
            temp = list()
            temp.append(batch)
            batch = temp

        batch_size = len(batch);
        wordids, wordcts = self.parse_doc_list(batch);
        
        gamma = self.e_step(batch_size, wordids, wordcts, True);
        #log_likelihood = self.log_likelihood(batch, gamma, True);
        return gamma;

    """
    Compute the aggregate digamma values, for phi update.
    """
    def compute_exp_weights_old(self):
        nu_1_over_nu_all = {};
        nu_2_over_nu_all = {};
        prod_nu_2_over_nu_all = {};
        exp_weights = {};
        exp_oov_weights = {};
        
        for k in xrange(self._number_of_topics):
            nu_1_over_nu_all[k] = self._nu_1[k] / (self._nu_1[k] + self._nu_2[k]);
            nu_2_over_nu_all[k] = self._nu_2[k] / (self._nu_1[k] + self._nu_2[k]);
            #assert(nu_1_over_nu_all.shape == (self._number_of_topics, self._truncation_size));
            #assert(nu_2_over_nu_all.shape == (self._number_of_topics, self._truncation_size));
        
            prod_nu_2_over_nu_all[k] = numpy.cumprod(nu_2_over_nu_all[k], axis=1);
            #assert(prod_nu_2_over_nu_all.shape == (self._number_of_topics, self._truncation_size));
            
            exp_weights[k] = numpy.hstack((numpy.ones((1, 1)), prod_nu_2_over_nu_all[k][:, :-1]));
            exp_weights[k] *= nu_1_over_nu_all[k];
            #assert(exp_weights.shape == (self._number_of_topics, self._truncation_size));
            #exp_oov_weights[k] = prod_nu_2_over_nu_all[k][:, -1][:, numpy.newaxis];
            exp_oov_weights[k] = prod_nu_2_over_nu_all[k][0, -1];
            #assert(exp_oov_weights.shape == (self._number_of_topics, 1));

        return exp_weights, exp_oov_weights;
