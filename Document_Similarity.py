import os
import random
import re
import string
import sys
import glob
import errno
import time

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import numpy as np
from scipy import spatial


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^:\n]*[:][^\n]*', '', text)
    a = string.punctuation
    for char in a:
        text = text.replace(char, "")
    text = ''.join([i for i in text if not i.isdigit()])
    text = text.strip()
    text = (nltk.word_tokenize(text))
    return text


def get_documents():
    graphic_files = read_graphics()

    # todo store one document in doc
    doc = graphic_files[0]
    # print(doc)

    # todo store 1 document from every folder other than comp.grpahics in other_docs
    t = time.time()
    temp_data = get_other_docs()
    print("Time taken to fetch docs = ", time.time() - t)
    train_data = []
    other_docs = []
    for folder in temp_data:
        other_docs = other_docs + [folder[0]]
        train_data = train_data + folder[1:]
    train_data = train_data + graphic_files[20:]

    # print(random.sample(other_docs))
    # todo store 19 docs from the same folder in same_docs
    same_docs = graphic_files[1:20]
    # print(random.sample(same_docs))

    # todo return the 3 tuple
    print(len(doc), len(other_docs), len(same_docs), len(train_data))
    return doc, other_docs, same_docs, train_data


def read_graphics(file_name='comp.graphics/*'):
    path = '/home/nishantsinha15/Documents/sem7/Natural Language Processing/Assignment 6/20_newsgroups/' + file_name
    files = glob.glob(path)
    ret = []
    print(path)
    for i, name in enumerate(files):  # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        try:
            with open(name, 'r') as f:  # No need to specify 'r': this is the default.
                try:
                    x = str(f.read())
                except:
                    print("Exception handled")
                    continue
                temp = preprocess(x)
                ret.append(temp)
        except IOError as exc:
            if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
                raise  # Propagate other kinds of IOError.
    return ret


def get_other_docs():
    path = '/home/nishantsinha15/Documents/sem7/Natural Language Processing/Assignment 6/20_newsgroups/'
    other_direc = [x[0] for x in os.walk(path)]
    other_direc = other_direc[1:]
    ret = []
    print(len(other_direc))
    for new_path in other_direc:
        if new_path == "/home/nishantsinha15/Documents/sem7/Natural Language Processing/Assignment 6/20_newsgroups/comp.graphics":
            continue
        files = glob.glob(new_path + "/*")
        this_folder = []
        print(new_path)
        for i, name in enumerate(files):  # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
            try:
                with open(name, 'r') as f:  # No need to specify 'r': this is the default.
                    try:
                        x = str(f.read())
                    except:
                        print("Exception handled")
                        continue
                    temp = preprocess(x)
                    this_folder.append(temp)
            except IOError as exc:
                if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
                    raise  # Propagate other kinds of IOError.
        ret.append(this_folder)

    return ret


def main():
    # todo train
    doc, other_docs, same_docs, train_data = get_documents()
    train_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_data)]
    # t = time.time()
    # model = Doc2Vec(train_documents, vector_size=50, window=2, min_count=1, workers=4)
    # print("Time taken to train model = ", time.time() - t)
    # model.save(fname_or_handle='mymodel')
    # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    fname = 'mymodel'
    model = Doc2Vec.load(fname)

    # todo get vectors for the required files
    doc = model.infer_vector(doc)
    print(doc)

    for i in range(len(other_docs)):
        other_docs[i] = model.infer_vector(other_docs[i])

    for i in range(len(same_docs)):
        same_docs[i] = model.infer_vector(same_docs[i])

    # todo get cosine similarity
    cosine_similarity(doc, other_docs, same_docs)

def cosine_similarity(doc, other_docs, same_docs):
    a = 0
    for vectors in other_docs:
        result = 1 - spatial.distance.cosine(doc, vectors)
        a += result
    a /= 19

    b = 0
    for vectors in same_docs:
        result = 1 - spatial.distance.cosine(doc, vectors)
        b += result
    b /= 19

    print("For same docs = ", b)
    print("For different docs = ", a)


if __name__ == '__main__':
    main()