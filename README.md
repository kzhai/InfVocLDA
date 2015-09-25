InfVocLDA
==========

InfVocLDA is a Latent Dirichlet Allocation topic modeling package based on Variational Bayesian learning approach under online settings, developed by the Cloud Computing Research Team in [University of Maryland, College Park] (http://www.umd.edu). You may find more details about this project on our papaer [Online Latent Dirichlet Allocation with Infinite Vocabulary] (http://kzhai.github.io/paper/2013_icml.pdf) appeared in ICML 2013.

Please download the latest version from our [GitHub repository](https://github.com/kzhai/InfVocLDA).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy and nltk. After downloading the source code packages, unzip the datasets to the 'input' directory. The package includes a few fundamental datasets --- ap, de-news and 20-newsgroup datasets.

Launch and Execute
----------

First, redirect to the source code directory

    cd InfVocLDA/src

To launch the online LDA with pre-defined vocabulary, run the following command

    python -m fixvoc.launch --input_directory=../input/ --output_directory=../output/ --corpus_name=20-news --number_of_topics=10 --number_of_documents=18600 --batch_size=100

To launch the online LDA with dynamic vocabulary, run the following command

    python -m infvoc.launch --input_directory=../input/ --output_directory=../output/ --corpus_name=de-news --truncation_level=4000 --number_of_topics=10 --number_of_documents=9800 --vocab_prune_interval=10 --batch_size=100 --alpha_beta=1000

Under any cirsumstances, you may also get help information and usage hints by running the following command

    python -m fixvoc.launch --help
    python -m infvoc.launch --help
