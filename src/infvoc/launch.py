#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import scipy.io;
import nltk;
import numpy;

from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

def retrieve_vocabulary(docs):
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    freq_dist = FreqDist();
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        ddict = dict()
        for word in words:
            if word not in stopwords.words('english'):
                freq_dist.inc(word);

    return freq_dist

def main():
    import option_parser;
    options = option_parser.parse_args();

    # parameter set 2
    assert(options.number_of_documents>0);
    number_of_documents = options.number_of_documents;
    assert(options.number_of_topics>0);
    number_of_topics = options.number_of_topics;
    assert(options.truncation_level>0);
    truncation_level = options.truncation_level;

    # parameter set 3
    assert(options.vocab_prune_interval>0);
    vocab_prune_interval = options.vocab_prune_interval;
    snapshot_interval = vocab_prune_interval;
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
    #assert(options.batch_size>0);
    batch_size = options.batch_size;
    #assert(number_of_documents % batch_size==0);
    online_iterations=number_of_documents/batch_size;
    if options.online_iterations>0:
        online_iterations=options.online_iterations;

    # parameter set 4
    assert(options.tau>=0);
    tau = options.tau;
    #assert(options.kappa>=0.5 and options.kappa<=1);
    assert(options.kappa>=0 and options.kappa<=1);
    kappa = options.kappa;
    if batch_size<=0:
        print "warning: running in batch mode..."
        kappa = 0;
    alpha_theta = 1.0/number_of_topics;
    if options.alpha_theta>0:
        alpha_theta=options.alpha_theta;
    assert(options.alpha_beta>0);
    alpha_beta = options.alpha_beta;
    
    # parameter set 5
    #heldout_data = options.heldout_data;

    # parameter set 1
    assert(options.corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);

    corpus_name = options.corpus_name;

    input_directory = options.input_directory;
    input_directory = os.path.join(input_directory, corpus_name);
        
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
     
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%b%d-%H%M%S")+"";
    suffix += "-D%d" % (number_of_documents);
    suffix += "-K%d" % (number_of_topics)
    suffix += "-T%d" % (truncation_level);
    suffix += "-P%d" % (vocab_prune_interval);
    suffix += "-I%d" % (snapshot_interval);
    suffix += "-B%d" % (batch_size);
    suffix += "-O%d" % (online_iterations);
    suffix += "-t%d" % (tau);
    suffix += "-k%g" % (kappa);
    suffix += "-at%g" % (alpha_theta);
    suffix += "-ab%g" % (alpha_beta);
    suffix += "/";
    '''
    suffix += "-D%d-K%d-T%d-P%d-S%d-B%d-O%d-t%d-k%g-at%g-ab%g/" % (number_of_documents,
                                                                   number_of_topics,
                                                                   truncation_level,
                                                                   vocab_prune_interval,
                                                                   snapshot_interval,
                                                                   batch_size,
                                                                   online_iterations,
                                                                   tau,
                                                                   kappa,
                                                                   alpha_theta,
                                                                   alpha_beta);
    '''
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
    
    dictionary_file = options.dictionary;
    if dictionary_file != None:
        dictionary_file = dictionary_file.strip();
        
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    options_output_file.write("dictionary_file=" + str(dictionary_file) + "\n");
    # parameter set 2
    options_output_file.write("number_of_documents=" + str(number_of_documents) + "\n");
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    options_output_file.write("truncation_level=" + str(truncation_level) + "\n");
    # parameter set 3
    options_output_file.write("vocab_prune_interval=" + str(vocab_prune_interval) + "\n");
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    options_output_file.write("batch_size=" + str(batch_size) + "\n");
    options_output_file.write("online_iterations=" + str(online_iterations) + "\n");
    # parameter set 4
    options_output_file.write("tau=" + str(tau) + "\n");
    options_output_file.write("kappa=" + str(kappa) + "\n");
    options_output_file.write("alpha_theta=" + str(alpha_theta) + "\n");
    options_output_file.write("alpha_beta=" + str(alpha_beta) + "\n");
    # parameter set 5    
    #options_output_file.write("heldout_data=" + str(heldout_data) + "\n");
    options_output_file.close()
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    print "dictionary_file=" + str(dictionary_file)
    # parameter set 2
    print "number_of_documents=" + str(number_of_documents)
    print "number_of_topics=" + str(number_of_topics)
    print "truncation_level=" + str(truncation_level)
    # parameter set 3
    print "vocab_prune_interval=" + str(vocab_prune_interval)
    print "snapshot_interval=" + str(snapshot_interval);
    print "batch_size=" + str(batch_size)
    print "online_iterations=" + str(online_iterations)
    # parameter set 4
    print "tau=" + str(tau)
    print "kappa=" + str(kappa)
    print "alpha_theta=" + str(alpha_theta)
    print "alpha_beta=" + str(alpha_beta)
    # parameter set 5
    #print "heldout_data=" + str(heldout_data)
    print "========== ========== ========== ========== =========="

    # Vocabulary
    #file = open(input_directory+'voc.dat', 'r');
    # Seed the vocabulary
    vocab = ['team'];

    # Documents
    train_docs = [];
    file = open(os.path.join(input_directory, 'doc.dat'), 'r');
    for line in file:
        train_docs.append(line.strip());
    print "successfully load all training documents..."

    import hybrid;
    olda = hybrid.Hybrid(3, 20)

    olda._initialize(vocab,
                     number_of_topics,
                     number_of_documents,
                     batch_size,
                     truncation_level,
                     alpha_theta,
                     alpha_beta,
                     tau,
                     kappa,
                     vocab_prune_interval,
                     True
                     );
    olda.export_beta(os.path.join(output_directory, 'exp_beta-0'), 100);
    
    document_topic_distribution = None;

    # Run until we've seen number_of_documents documents. (Feel free to interrupt *much* sooner than this.)
    for iteration in xrange(online_iterations):
        if batch_size<=0:
            docset = train_docs;
        else:
            docset = train_docs[(batch_size * iteration) % len(train_docs) : (batch_size * (iteration+1) - 1) % len(train_docs) + 1];
            print "select documents from %d to %d" % ((batch_size * iteration) % (number_of_documents), (batch_size * (iteration+1) - 1) % number_of_documents + 1)

        clock = time.time();

        batch_gamma = olda.learning(docset);
        
        if (olda._counter % snapshot_interval == 0):
            olda.export_beta(os.path.join(output_directory, 'exp_beta-' + str(olda._counter)), 50);

        if document_topic_distribution is None:
            document_topic_distribution = batch_gamma;
        else:
            document_topic_distribution = numpy.vstack((document_topic_distribution, batch_gamma));
        
        clock = time.time()-clock;
        print "vocabulary size = %s" % (olda._truncation_size);
        print 'training iteration %d finished in %f seconds: epsilon = %f' % (olda._counter, clock, olda._epsilon);

    gamma_path = os.path.join(output_directory, "gamma.txt");
    numpy.savetxt(gamma_path, document_topic_distribution);

    '''
    olda.export_word_index(os.path.join(output_directory, olda._index_title + str(olda._counter)));
    
    if olda._word_trace!=None:
        for index in xrange(len(olda._word_trace)):
            if numpy.min(olda._word_trace[index]) > 50:
                continue;
            ranking_statistics_path = os.path.join(output_directory, olda._ranking_statistics_title + olda._index_to_word[index] + str(-numpy.min(olda._word_trace[index])));
            assert olda._word_trace[index].dtype=='int32';
            numpy.savetxt(ranking_statistics_path, olda._word_trace[index], '%d');
    '''
    
if __name__ == '__main__':
    main()