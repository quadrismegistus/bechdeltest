from bechdeltest import *

# functions
def get_convokit_corpus(corpus_name):
    from convokit import Corpus, download
    return Corpus(download(corpus_name))