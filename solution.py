import spacy


def lemma_pos(x="This is an apple. I think it is tasty. Nishant is awesome"):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(x)
    for token in doc:
        print(token.text, "\nlemma = ", token.lemma_, "\nPOS tag = ", token.tag_)
        print()


def ner(x='Apple is looking at buying U.K. startup for $1 billion'):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(x)
    for ent in doc.ents:
        print(ent.text, ent.label_)


def similarity(a, b):
    nlp = spacy.load('en_core_web_sm')  # A larger model would perform better
    a = nlp(a)
    b = nlp(b)
    for token in a:
        for token2 in b:
            print(token.similarity(token2))


def word2vec():
    import gensim, time
    t = time.time()
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print("Loading took ", time.time() - t ," seconds.")
    print(model.most_similar(positive=['woman', 'king'], negative=['man']))


word2vec()


'''    
similarity('cat', 'dog')
similarity('cat', 'tiger')
similarity('cat', 'pussy')
similarity('cat', 'banana')
'''
