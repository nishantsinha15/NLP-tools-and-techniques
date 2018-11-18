import gensim, time

def word2vec():
    t = time.time()
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print("Loading took ", time.time() - t ," seconds.")