import random
import string
import nltk
from itertools import chain
from math import log, pow
import math;
import numpy;
import re;
import sys
import time;

from nltk.probability import ConditionalProbDist, ConditionalFreqDist, MLEProbDist, LaplaceProbDist, SimpleGoodTuringProbDist
#GoodTuringProbDist#, 
from nltk.util import ngrams

#from nltk.model.api import ModelI

#random.seed(99999999);

def _estimator(fdist, bins):
    """
    Default estimator function using a SimpleGoodTuringProbDist.
    """
    # can't be an instance method of NgramModel as they 
    # can't be pickled either.
    
    #return GoodTuringProbDist(fdist);
    return SimpleGoodTuringProbDist(fdist);

class NcharModel(ModelI):
    """
    A processing interface for assigning a probability to the next word.
    """
    
    def __init__(self,
                 n,
                 train,
                 smoothing=1e9,
                 #lagrangian_parameter=1.,
                 #estimator=None,
                 maximum_length=20,
                 minimum_length=3,
                 char_set=string.lowercase + string.punctuation + string.digits,
                 #char_set=string.lowercase,
                 patch_char='#'):
        """
        Creates an nchar language model to capture patterns in n consecutive
        words of training text.  An estimator smooths the probabilities derived
        from the text and may allow generation of ngrams not seen during
        training.

        @param n: the order of the language model (nchar size)
        @type n: C{int}
        @param train: the training text
        @type train: C{list} of C{string}
        @param estimator: a function for generating a probability distribution
        @type estimator: a function that takes a C{ConditionalFreqDist} and
              returns a C{ConditionalProbDist}
        """

        self._smoothing = smoothing;
        #self.lagrangian_parameter = lagrangian_parameter;

        self._n = n

        self._maximum_length = maximum_length;
        self._minimum_length = minimum_length;
        self._char_set = char_set;
        
        #estimator = lambda fdist, bins: nltk.probability.WittenBellProbDist(fdist, len(char_set));
        estimator = lambda fdist, bins: nltk.probability.LidstoneProbDist(fdist, self._smoothing, len(self._char_set)+1);
        #estimator = lambda fdist, bins: nltk.probability.LidstoneProbDist(fdist, 1e-9, len(self._char_set));
        #estimator = lambda fdist, bins: nltk.probability.GoodTuringProbDist(fdist, len(self._char_set));
        #estimator = lambda fdist, bins: nltk.probability.SimpleGoodTuringProbDist(fdist, len(self._char_set));

        cfd = ConditionalFreqDist()
        self._ngrams = set()
        self._patch_char = patch_char;
        self._prefix = (self._patch_char,) * (n - 1)
        
        length = nltk.probability.FreqDist();
        word_freq_dist = nltk.probability.FreqDist();
        char_list = [];
        for word in train:
            word = word.strip().lower();
            if len(word)<self._minimum_length or len(word)>self._maximum_length:
                continue;
            length.inc(len(word));
            word_freq_dist.inc(word, 1);
            char_list.extend(self._prefix);
            char_list.extend([char for char in word if char in self._char_set]);
        self._length = nltk.probability.WittenBellProbDist(length, length.B()+1);
        #self._length = nltk.probability.WittenBellProbDist(length, self._maximum_length);
        
        #context_freq_dist = nltk.probability.FreqDist();
        #for nchar in ingrams(chain(self._prefix, train), n):
        for nchar in ngrams(char_list, n):
            self._ngrams.add(nchar)
            context = tuple(nchar[:-1])
            #context_freq_dist.inc(context);
            token = nchar[-1]
            cfd[context].inc(token)
        #self._context = nltk.probability.WittenBellProbDist(context_freq_dist, len(self._char_set)**(n-1)+1);

        '''
        if n==3:
            cond = 0;
            for x in self._char_set:
                for y in self._char_set:
                    print (x, y), context_freq_dist[(x, y)], self._context.prob((x, y));
                    cond += self._context.prob((x, y));
            print 'cond is', cond
        '''
        
        #self._model = ConditionalProbDist(cfd, estimator, len(cfd));
        #print self._char_set;
        self._model = ConditionalProbDist(cfd, estimator, len(self._char_set) ** (n - 1));

        #========== ========== ========== ========== ========== ========== ========== ========== ========== ========== ========== ========== ========== ==========
        '''
        consonant_freq_dist = nltk.probability.FreqDist();
        consonant_condition_freq_dist = nltk.probability.ConditionalFreqDist();
        for word in train:
            #word = re.sub(r'aeiou', ' ', word);
            word = word[0] + re.sub('aeiouy', ' ', word[1:]);
            
            consonant_list = word.split();
            #consonant_list = ['#', '#'] + consonant_list;
            for temp in consonant_list:
                consonant_freq_dist.inc(temp, 1);
                
        consonant_freq_dist.plot()
        '''
        #========== ========== ========== ========== ========== ========== ========== ========== ========== ========== ========== ========== ========== ==========        
        word_prob_dist = nltk.probability.MLEProbDist(word_freq_dist);

        word_model_empirical_frequency = numpy.zeros((1, self._maximum_length - self._minimum_length + 1)) + 1e-300;
        word_model_square = numpy.zeros((1, self._maximum_length - self._minimum_length + 1)) + 1e-300;
        
        #word_model_empirical_frequency_old = numpy.zeros((1, self._maximum_length - self._minimum_length + 1));
        #word_model_square_old = numpy.zeros((1, self._maximum_length - self._minimum_length + 1));
        
        total_outcomes = 0;
        for x in xrange(self._minimum_length, self._maximum_length+1):
            total_outcomes += len(self._char_set) ** x;

        for word in word_freq_dist.keys():
            word_model_empirical_frequency[0, len(word)-self._minimum_length] += word_prob_dist.prob(word) * self.probability_without_length(word);
            #word_model_empirical_frequency[0, len(word)-self._minimum_length] += 1.0/total_outcomes * self.probability_without_length(word);
            word_model_square[0, len(word)-self._minimum_length] += self.probability_without_length(word) ** 2;
            
            #word_model_empirical_frequency_old[0, len(word)-self._minimum_length] += word_prob_dist.prob(word) * self.probability_without_length(word);
            #word_model_square_old[0, len(word)-self._minimum_length] += self.probability_without_length(word) ** 2;
        
        #print "alpha is", 2 * (1-numpy.sum(word_model_empirical_frequency / word_model_square))/numpy.sum(1.0/word_model_square)
        #print word_model_empirical_frequency, word_model_square

        #sum_word_model_square_inverse = numpy.sum(1.0 / word_model_square);
        #sum_word_model_empirical_frequency_over_word_model_square = numpy.sum(word_model_empirical_frequency / word_model_square);
        #self._multinomial_length = (word_model_empirical_frequency * sum_word_model_square_inverse - sum_word_model_empirical_frequency_over_word_model_square + 1) / (word_model_square * sum_word_model_square_inverse);
        #print sum_word_model_square_inverse, sum_word_model_empirical_frequency_over_word_model_square;
        #print self._multinomial_length, numpy.sum(self._multinomial_length);
            
        if True:
            lagrangian_parameter = 2 * (1-numpy.sum(word_model_empirical_frequency / word_model_square))/numpy.sum(1.0/word_model_square)
        else:
            lagrangian_parameter = 1.;
        #print "lagrangian parameter is", lagrangian_parameter
        self._multinomial_length = (word_model_empirical_frequency - lagrangian_parameter / 2) / word_model_square;
        self._multinomial_length /= numpy.sum(self._multinomial_length);
        
        #print self._multinomial_length, numpy.sum(self._multinomial_length);
        assert numpy.all(self._multinomial_length>=0), self._multinomial_length;

        # recursively construct the lower-order models
        if n > 1:
            self._backoff = NcharModel(n-1, train, self._smoothing, maximum_length,
                 minimum_length, self._char_set, self._patch_char);
        
    def probability(self, word):
        # This is the naive way to write compute probability without backoff
        geometric_mean=False;
        
        if len(word)<self._minimum_length or len(word)>self._maximum_length:
            return 0;

        if geometric_mean:
            prob = 1.0;
        else:
            #prob = self._length.prob(len(word));
            prob = self._multinomial_length[0, len(word)-self._minimum_length];
        word = [char for char in word];
        word = tuple(word);
        word = self._prefix + word;
        for i in xrange(len(word) - 1, 1, -1):
            prob *= self._model[word[i - (self._n - 1):i]].prob(word[i]);
        if geometric_mean:
            prob = pow(prob, 1.0/(len(word) - len(self._prefix)));

        return prob;
        
        '''
        # This is the old version of the code, using backoff models
        prob = 1;
        for i in xrange(len(word)-1, -1, -1):
            if i-(self._n-1)<0:
                #print word[i], "|", word[:i]
                prob *= self.prob(word[i], word[:i]);
            else:
                #print word[i], '|', word[i-(self._n-1):i];
                prob *= self.prob(word[i], word[i-(self._n-1):i]);
                
        return prob;
        '''

    def probability_without_length(self, word):
        # This is the naive way to write compute probability without backoff
        prob = 1.0;
        word = [char for char in word];
        word = tuple(word);
        word = self._prefix + word;
        for i in xrange(len(word) - 1, 1, -1):
            prob *= self._model[word[i - (self._n - 1):i]].prob(word[i]);

        return prob;
        
    '''
    Katz Backoff probability
    @deprecated
    '''
    def prob(self, charactor, context):
        """
        Evaluate the probability of this char in this context.
        """
        # This is the bug-fixed code from the web.
        context = [char for char in context];
        context = tuple(context);
        context = self._prefix + context;
        context = context[-(self._n - 1):];
        if (context + (charactor,) in self._ngrams) or (self._n == 1):
            return self._model[context].prob(charactor)
            #return self[context].prob(charactor)
        else:
            #print 'backoff', charactor, context[1:], self._backoff.prob(charactor, context[1:]);
            return self._alpha(context) * self._backoff.prob(charactor, context[1:])
        
        '''
        # This is the original code.
        context = tuple(context)
        if context + (charactor,) in self._ngrams:
            return self[context].prob(charactor)
        elif self._n > 1:
            return self._alpha(context) * self._backoff.prob(charactor, context[1:])
        else:
            raise RuntimeError("No probability mass assigned to charactor %s in "
                               "context %s" % (charactor, ' '.join(context)))
        '''
        
    def _alpha(self, tokens):
        return self._beta(tokens) / self._backoff._beta(tokens[1:])

    def _beta(self, tokens):
        #print "tokens is ", tokens
        #print self
        #print tokens in self
        if len(tokens) > 0 and tokens in self:
            return self[tokens].discount()
        else:
            return 1

    def logprob(self, word, context):
        """
        Evaluate the (negative) log probability of this word in this context.
        """
        return - log(self.prob(word, context), 2)

    def choose_random_word(self, context):
        '''Randomly select a word that is likely to appear in this context.'''
        return self.generate(1, context)[-1]

    # NB, this will always start with same word since model
    # is trained on a single text
    def generate(self, num_words, context=()):
        '''Generate random text based on the language model.'''
        text = list(context)
        for i in range(num_words):
            text.append(self._generate_one(text))
        return text

    def _generate_one(self, context):
        context = (self._prefix + tuple(context))[-self._n + 1:]
        # print "Context (%d): <%s>" % (self._n, ','.join(context))
        if context in self:
            return self[context].generate()
        elif self._n > 1:
            return self._backoff._generate_one(context[1:])
        else:
            return '.'

    def entropy(self, text):
        """
        Evaluate the total entropy of a text with respect to the model.
        This is the sum of the log probability of each word in the message.
        """

        e = 0.0
        for i in range(self._n - 1, len(text)):
            context = tuple(text[i - self._n + 1:i])
            token = text[i]
            e += self.logprob(token, context)
        return e

    def __contains__(self, item):
        return tuple(item) in self._model

    def __getitem__(self, item):
        return self._model[tuple(item)]

    def __repr__(self):
        return '<NgramModel with %d %d-grams>' % (len(self._ngrams), self._n)

def demo():
    import string;
    
    char_list = string.lowercase + string.digits
    #print char_list;

    import re;            
    from nltk.corpus import brown
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer();

    #'''
    if True:
        file = '../data/words_english.txt';
        #char_list=['a', 'b'];
        #file = '../data/test_word_model.txt';
        words = [];
        for line in open(file, 'r'):
            line = line.strip();
            if len(line) <= 0:
                continue;
            words.append(stemmer.stem(line));
    else:
        words = [];
        for word in brown.words():
            word = word.lower();
            word = re.sub(r'-', ' ', word);
            word = re.sub(r'[^a-z0-9 ]', '', word);
            word = re.sub(r' +', ' ', word);
            if len(word)==0:
                continue;
            if word in nltk.corpus.stopwords.words('english'):
                continue;
            
            words.append(stemmer.stem(word));
    #'''

    '''
    #words = [];
    words = nltk.probability.FreqDist();
    for word in brown.words():
        word = word.lower();
        word = re.sub(r'-', ' ', word);
        word = re.sub(r'[^a-z0-9 ]', '', word);
        word = re.sub(r' +', ' ', word);
        if len(word)==0:
            continue;
        if word in nltk.corpus.stopwords.words('english'):
            continue;
        
        #words.append(stemmer.stem(word));
        words.inc(stemmer.stem(word), 1)
    #print words
    
    #print len(words);
    temp_words = [];
    for word in words.keys()[-len(words)/2:]:
        temp_words.append(word);
    words = temp_words
    #print len(words);
    '''
    
    wm = NcharModel(3, words, 1e9, 20, 3, char_list);
    
    clock = time.time();
    input = open('../input/words_english.txt', 'r')
    for line in input:
        wm.probability(line.strip());
    print time.time()-clock;

    '''
    smoothing = 1e10;
    while (True):
        wm = NcharModel(3, words, smoothing, 20, 3, char_list);
        
        output_file = open('../output/nchar-dict-3/nchar-'+str(smoothing), 'w');
        for line in open('../input/nyt/voc.dat', 'r'):
            line = line.strip();
            line = line.split()[0];
            output_file.write(line + "\t" + str(wm.probability(line)) + "\n");
        
        smoothing /= 10;
        if smoothing==1:
            break;
    '''

    '''
    test_word_list = ['python', 'google', 'reduce', 'hadoop', 'long', 'blog', 'weibo', 'ke', 'hu', 'zhaike', 'yuening', 'ynhu']; 
    for word in test_word_list:
        print word, "with probability", wm.probability(word);
        print line, wm.probability_over_permutation(line);
    '''

if __name__ == '__main__':
    demo();