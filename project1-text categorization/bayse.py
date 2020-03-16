import nltk
import parse
import math
import collections
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from parse import parse
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
import numpy as np
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer('english')
label_word_dict = {}
label_doc_word_dict = {}
label_numDiff = {}
label_numTot = {}
label_freq = {}
tot_label = 0
def parsing(path):
	words = [stemmer.stem(w) for w in tokenizer.tokenize(open(path, "r").read().lower())]
	#words = [word for word in words if word not in stopwords.words('english')]
	filtered_words = [word for word in words if word not in stopwords.words('english')]
	#print(path)
	return filtered_words

def training(input):
    doc = open(input, "r")
    for line in doc:
    	path,label = parse.parse("{} {}",line).fixed
    	if label not in label_word_dict.keys():
    		label_word_dict[label] = []
    		label_freq[label] = 1

    	if label not in label_doc_word_dict.keys():
    		label_doc_word_dict[label] = {}
    	
    	label_freq[label] += 1
    	words = parsing(path)
    	label_word_dict[label] = label_word_dict[label]+words
    	label_doc_word_dict[label][path] = collections.Counter(words)


    for label in label_word_dict.keys():
    	label_numTot[label] = len(label_word_dict[label])
    	label_word_dict[label] = collections.Counter(label_word_dict[label])
    	label_numDiff[label] = len(label_word_dict[label])

    #print(label_word_dict)
    #print(label_freq)
    #print(label_numDiff)
    #print(label_numTot)
    #print(len(label_word_dict))
    #print(label_doc_word_dict)
    tot_label = sum(label_freq.values())
    for label in label_freq:
    	label_freq[label] = label_freq[label] / tot_label
    #print(label_freq)

def get_percent(count,label):
	k = 3.5
	total = label_numTot[label]
	distinct = label_numDiff[label]
	res = 0
	unknown = 0
	#print(count.keys())
	#print(label_word_dict[label].keys())
	for word in count.keys():
		if word not in label_word_dict[label].keys():
			#print("did not find the word " + word)
			unknown += count[word]

	#print(unknown)
	for word in count.keys():
		if word in label_word_dict[label]:
			percent = (label_word_dict[label][word] + k * unknown) * count[word] / (total + k*unknown*distinct)
			#percent = (label_word_dict[label][word] + k * unknown) * count[word] / (total)

		else:
			#print('did not find the word, need to do smoothing')
			#percent = count[word]*k / (total)
			percent = (count[word]*k) / (total + k*unknown*distinct)

		res += percent

	#print(math.log10(label_freq[label]/ sum(label_freq.values())))
	#print(res)
	#print(label_freq[label])
	res = math.log10(res) #+ math.log10(label_freq[label])
	#res = res + label_freq[label] * tot_label * 10000
	return res

def compute_percent(count,label):
	total = len(label_doc_word_dict[label])
	doc_words_dict = label_doc_word_dict[label]
	num_doc = 0
	percent = 0
	for path in doc_words_dict.keys():
		for word in count.keys():
			if word in doc_words_dict[path].keys():
				num_doc += 1
				break

	for word in count.keys():
		for doc in doc_words_dict.keys():
			if word in doc_words_dict[doc].keys():
				num_doc += 1

		percent += num_doc / total
		num_doc = 0

	#print(label)
	#print('total docs: ' + str(total))
	#print('percent: ',percent)
	#print('label freq: ' + str(label_freq[label]))
	#print('\n')
	return math.log10(percent) + math.log10(label_freq[label])




def findmax(label_prob):
	res = ''
	max = -100000000000
	for label in label_prob:
		if label_prob[label] > max:
			max = label_prob[label]
			res = label

	return res


def testing(test_doc,outputfile):
	c = 0
	test_doc = open(test_doc,"r").read().split('\n')
	outputfile = open(outputfile,"wb")
	for line in test_doc:
		c += 1
		label_prob = {}
		words = parsing(line)
		count = collections.Counter(words)
		#print(count)
		for label in label_freq.keys():
			percent = get_percent(count,label)
			label_prob[label] = percent

		result = findmax(label_prob)
		#print(label_prob)
		#print(result)
		outputfile.write(f'{line} {result}\n'.encode('ascii'))
		#if c == 3:
		#	break

if __name__ == '__main__':
	train_doc = input('input training file: ')
	#train_doc = 'corpus1_train.labels'
	print('training...')
	training(train_doc)
	print('training complete')
	test_doc = input('input testing file: ')
	#test_doc = 'corpus1_test.list'
	#outputfile = '123'
	outputfile = input('output file: ')
	print('testing...')
	testing(test_doc,outputfile)
	print('file was generated')
