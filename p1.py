import math
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np

def parsing(path):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer('english')
    doc_words = [stemmer.stem(w) for w in tokenizer.tokenize(open(path, "r").read().lower())]
    return doc_words

def get_tfidf(doc_term_freq, doc_freq, num_docs):
    msum = 0.0
    max_term_freq = max(doc_term_freq.values())
    per_doc_tfidf = {}
    for word in doc_term_freq:
        word_term_freq = 0.4 + 0.6 * (doc_term_freq[word] / max_term_freq)
        if word in doc_freq:
            idf = math.log10(num_docs / doc_freq[word])
            tfidf = word_term_freq * idf
            per_doc_tfidf[word] = tfidf
            msum += tfidf * tfidf

    norm_factor = 1.0 / math.sqrt(msum)
    for word in per_doc_tfidf:
        temp = per_doc_tfidf[word]
        per_doc_tfidf[word] = temp * norm_factor

    return per_doc_tfidf

def update_freq(path, word_freq, doc_freq, train=True):
    freq_count = Counter(parsing(path))
    word_freq[path] = freq_count
    if train:
        for e in freq_count:
            if e in doc_freq:
                temp = doc_freq[e]
                doc_freq[e] = temp + 1
            else:
                doc_freq[e] = 1
        return doc_freq, word_freq
    return word_freq


def get_cen(c, doc_label, doc_tfidf, num_docs):
    beta = 0.11
    alpha = 4.1
    cat_doc_count = list(doc_label.values()).count(c)
    pos_factor = alpha / cat_doc_count
    neg_factor = beta / (num_docs - cat_doc_count)
    centroid = {}
    pos_sum = {}
    neg_sum = {}

    for doc in doc_label:
        per_doc_tfidf = doc_tfidf[doc]
        if doc_label[doc] == c:
            for word in per_doc_tfidf:
                if word in pos_sum:
                    temp = pos_sum[word]
                    pos_sum[word] = temp + (per_doc_tfidf[word] * pos_factor)
                else:
                    pos_sum[word] = per_doc_tfidf[word] * pos_factor
        else:
            for word in per_doc_tfidf:
                if word in neg_sum:
                    temp = neg_sum[word]
                    neg_sum[word] = temp + (per_doc_tfidf[word] * neg_factor)
                else:
                    neg_sum[word] = per_doc_tfidf[word] * neg_factor

    mag_sum = 0.0
    for word in pos_sum:
        if word in neg_sum:
            val = pos_sum[word] - neg_sum[word]
            centroid[word] = val
            mag_sum += val * val
        else:
            centroid[word] = pos_sum[word]
            mag_sum += pos_sum[word] * pos_sum[word]

    norm_factor = 1.0 / math.sqrt(mag_sum)
    for word in centroid:
        temp = centroid[word]
        centroid[word] = temp * norm_factor

    return centroid

def train(training_file):
    training_list = open(training_file, "r").read().split("\n")
    num_docs = len(training_list)
    doc_category = {}
    doc_count = {}
    term_freq = {}
    doc_tfidf = {}
    category_centroids = {}

    for doc_full in training_list:
        split_doc = doc_full.split()
        if split_doc:
            path = split_doc[0]
            doc_type = split_doc[1]
            doc_category[path] = doc_type
            update_freq(path, term_freq, doc_count)

    for doc in term_freq:
        doc_tfidf[doc] = get_tfidf(term_freq[doc], doc_count, num_docs)

    for c in set(doc_category.values()):
        category_centroids[c] = get_cen(c, doc_category, doc_tfidf, num_docs)

    return category_centroids, doc_count, len(training_list)

def find(test_vector, l):
    res = 0
    for e in test_vector:
        if e in l:
            res += test_vector[e] * l[e]
    return res


def testing(testing_file, centroids, doc_freq, num_docs, output_file):
    test_list = open(testing_file, "r").read().split("\n")
    output = open(output_file, "wb")
    for path in test_list:
        #print(path)
        #break
        if path:
            path = path.rstrip()
            term_freq = update_freq(path, {}, {}, train=False)
            tfidf = get_tfidf(term_freq[path], doc_freq, num_docs)
            cosine_score = 0.0
            label = None
            for c in centroids:
                result = find(tfidf, centroids[c])
                if result > cosine_score:
                    cosine_score = result
                    label = c
        output.write(f'{path} {label}\n'.encode('ascii'))
        #output.write(path)
        #output.write(label)
    output.close()


if __name__ == '__main__':
    train_doc = input("Enter the file of labeled training documents: ")
    test_doc = input("Enter the file of unlabeled test documents: ")
    print("training...")
    centroids, doc_freq, num_docs = train(train_doc)
    output = input("Enter the output file: ")
    print("testing...")
    testing(test_doc, centroids, doc_freq, num_docs, output)
    print("file was generated successfully")









