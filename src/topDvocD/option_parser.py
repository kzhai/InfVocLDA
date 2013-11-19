import optparse;

delimiter = '-';

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        corpus_name=None,
                        dictionary=None,
                        
                        # parameter set 2
                        number_of_topics=25,
                        number_of_documents=-1,

                        # parameter set 3
                        snapshot_interval=10,
                        batch_size=-1,
                        online_iterations=-1,
                        
                        # parameter set 4
                        kappa=0.6,
                        tau=64.0,
                        alpha_theta=-1,
                        alpha_eta=-1,
                        
                        # parameter set 5
                        hybrid_mode=False,
                        hash_oov_words=False
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      help="the corpus name [None]")
    parser.add_option("--dictionary", type="string", dest="dictionary",
                      help="the dictionary file [None]")
    
    # parameter set 2
    parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      help="second level truncation [25]");
    parser.add_option("--number_of_documents", type="int", dest="number_of_documents",
                      help="number of documents [-1]");

    # parameter set 3
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [10]");
    parser.add_option("--batch_size", type="int", dest="batch_size",
                      help="batch size [-1 in batch mode]");
    parser.add_option("--online_iterations", type="int", dest="online_iterations",
                      help="max iteration to run training [number_of_documents/batch_size]");
                      
    # parameter set 4
    parser.add_option("--kappa", type="float", dest="kappa",
                      help="learning rate [0.6]")
    parser.add_option("--tau", type="float", dest="tau",
                      help="learning inertia [64.0]")
    parser.add_option("--alpha_theta", type="float", dest="alpha_theta",
                      help="hyper-parameter for Dirichlet distribution of topics [1.0/number_of_topics]")
    parser.add_option("--alpha_eta", type="float", dest="alpha_eta",
                      help="hyper-parameter for Dirichlet distribution of vocabulary [1.0/number_of_vocabulary]")
    
    # parameter set 5
    parser.add_option("--hybrid_mode", action='store_true', default=False, dest="hybrid_mode",
                      help="local inference in hybrid mode [False]")
    parser.add_option("--hash_oov_words", action='store_true', default=False, dest="hash_oov_words",
                      help="hash out-of-vocabulary words to run this model in pseudo infinite vocabulary mode[False]")
    
    (options, args) = parser.parse_args();
    return options;