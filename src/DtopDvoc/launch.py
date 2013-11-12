import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

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

    # parameter set 3
    assert(options.snapshot_interval>0);
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
    
    # parameter set 5
    hybrid_mode = options.hybrid_mode;
    hash_oov_words = options.hash_oov_words;

    # parameter set 1
    assert(options.corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);

    corpus_name = options.corpus_name;

    input_directory = options.input_directory;
    #if not input_directory.endswith('/'):
        #input_directory += '/';
    input_directory = os.path.join(input_directory, corpus_name);
    #input_directory += corpus_name+'/';
        
    output_directory = options.output_directory;
    #if not output_directory.endswith('/'):
        #output_directory += '/';
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    #output_directory += corpus_name+'/';
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
     
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%b%d-%H%M%S")+"";
    suffix += "-D%d-K%d-S%d-B%d-O%d-t%d-k%g-at%g-%s-%s/" % (number_of_documents,
                                                            number_of_topics,
                                                            snapshot_interval,
                                                            batch_size,
                                                            online_iterations,
                                                            tau,
                                                            kappa,
                                                            alpha_theta,
                                                            hybrid_mode,
                                                            hash_oov_words);
    output_directory = os.path.join(output_directory, suffix);
                                                           

    os.mkdir(os.path.abspath(output_directory));
    
    dict_file = options.dictionary;
        
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    options_output_file.write("dictionary_file=" + str(dict_file) + "\n");
    # parameter set 2
    options_output_file.write("number_of_documents=" + str(number_of_documents) + "\n");
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    options_output_file.write("batch_size=" + str(batch_size) + "\n");
    options_output_file.write("online_iterations=" + str(online_iterations) + "\n");
    # parameter set 4
    options_output_file.write("tau=" + str(tau) + "\n");
    options_output_file.write("kappa=" + str(kappa) + "\n");
    options_output_file.write("alpha_theta=" + str(alpha_theta) + "\n");
    # parameter set 5
    options_output_file.write("hybrid_mode=" + str(hybrid_mode) + "\n");
    options_output_file.write("hash_oov_words=%s\n" % hash_oov_words);
    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    print "dictionary file=" + str(dict_file)
    # parameter set 2
    print "number_of_documents=" + str(number_of_documents)
    print "number_of_topics=" + str(number_of_topics)
    # parameter set 3
    print "snapshot_interval=" + str(snapshot_interval);
    print "batch_size=" + str(batch_size)
    print "online_iterations=" + str(online_iterations)
    # parameter set 4
    print "tau=" + str(tau)
    print "kappa=" + str(kappa)
    print "alpha_theta=" + str(alpha_theta)
    # parameter set 5
    print "hybrid_mode=" + str(hybrid_mode)
    print "hash_oov_words=%s" % (hash_oov_words)
    print "========== ========== ========== ========== =========="

    # Vocabulary
    if dict_file==None:
        vocab = [];
        line_count = 0;
        for line in open(os.path.join(input_directory, 'voc.dat'), 'r'):
            vocab.append(line.strip().split()[0]);
            line_count += 1
            if line_count>=batch_size:
                break;
        vocab = list(set(vocab));
        print "successfully load all the words from first epoch..."
    else:
        input_file = open(dict_file, 'r');
        vocab = [];
        for line in input_file:
            vocab.append(line.strip().split()[0]);
        print "successfully load all the dictionary words..."

    # Documents
    train_docs = [];
    input_file = open(os.path.join(input_directory, 'doc.dat'), 'r');
    for line in input_file:
        train_docs.append(line.strip());
    print "successfully load all training documents..."

    eta = 1./number_of_topics;

    if hybrid_mode:
        import hybrid;
        olda = hybrid.Hybrid(hash_oov_words);
    else:
        import variational;
        olda = variational.Variational(hash_oov_words);
        
    olda._initialize(vocab, number_of_topics, number_of_documents, alpha_theta, eta, tau, kappa);

    olda.export_beta(os.path.join(output_directory, 'exp_beta-0'), 50);

    document_topic_distribution = None;

    for iteration in xrange(online_iterations):
        if batch_size<=0:
            docset = train_docs;
        else:
            docset = train_docs[(batch_size * iteration) % len(train_docs) : (batch_size * (iteration+1) - 1) % len(train_docs) + 1];
            print "select documents from %d to %d" % ((batch_size * iteration) % (number_of_documents), (batch_size * (iteration+1) - 1) % number_of_documents + 1)

        clock = time.time();
        
        batch_gamma, elbo = olda.learning(docset)

        if document_topic_distribution==None:
            document_topic_distribution = batch_gamma;
        else:
            document_topic_distribution = numpy.vstack((document_topic_distribution, batch_gamma));
            
        clock = time.time()-clock;
        print 'training iteration %d finished in %f seconds: epsilon = %f' % (olda._counter, clock, olda._epsilon);

        # Save lambda, the parameters to the variational distributions over topics, and batch_gamma, the parameters to the variational distributions over topic weights for the articles analyzed in the last iteration.
        #if ((olda._counter+1) % snapshot_interval == 0):
            #olda.export_beta(output_directory + 'exp_beta-' + str(olda._counter+1));
        if (olda._counter % snapshot_interval == 0):
            olda.export_beta(os.path.join(output_directory, 'exp_beta-' + str(olda._counter)), 50);
    
    gamma_path = os.path.join(output_directory, 'gamma.txt');
    numpy.savetxt(gamma_path, document_topic_distribution);
    
if __name__ == '__main__':
    main()