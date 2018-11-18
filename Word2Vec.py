import gensim, time

t = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print("Loading took ", time.time() - t ," seconds.")


print(model.most_similar(positive=['Delhi', 'China'], negative=['India'], topn=10))

print(model.most_similar(positive=['Isro', 'USA'], negative=['India'], topn=10))
