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
label_numDiff = {}
label_numTot = {}
label_freq = {}
tot_label = 0
def parsing(path):
	words = [stemmer.stem(w) for w in tokenizer.tokenize(open(path, "r").read().lower())]
	#print(path)
	return words

def training(input):
    doc = open(input, "r")
    for line in doc:
    	path,label = parse.parse("{} {}",line).fixed
    	if label not in label_word_dict.keys():
    		label_word_dict[label] = []
    		label_freq[label] = 1

    	label_freq[label] += 1
    	words = parsing(path)
    	label_word_dict[label] = label_word_dict[label]+words

    #print(label_word_dict)

    for label in label_word_dict.keys():
    	label_numTot[label] = len(label_word_dict[label])
    	label_word_dict[label] = collections.Counter(label_word_dict[label])
    	label_numDiff[label] = len(label_word_dict[label])

    #print(label_word_dict)
    #print(label_freq)
    #print(label_numDiff)
    #print(label_numTot)
    #print(len(label_word_dict))
    tot_label = sum(label_freq.values())
    for label in label_freq:
    	label_freq[label] = label_freq[label] / tot_label
    #print(label_freq)

def get_percent(count,label):
	k = 0.75
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
		else:
			#print('did not find the word, need to do smoothing')
			percent = count[word]*k / (total + k*unknown*distinct)

		res += percent

	#print(math.log10(label_freq[label]/ sum(label_freq.values())))
	#print(res)
	#print(label_freq[label])
	res = math.log10(res) + math.log10(label_freq[label])*0.7
	#res = res + label_freq[label] * tot_label * 10000
	return res


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
		print(label_prob)
		#print(result)
		outputfile.write(f'{line} {result}\n'.encode('ascii'))
		#if c == 3:
		#	break

if __name__ == '__main__':
	#train_doc = input('input file: ')
	train_doc = 'corpus1_train.labels'
	training(train_doc)
	print('training complete')
	#test_doc = input('test file: ')
	test_doc = 'corpus1_test.list'
	outputfile = '123'
	#outputfile = input('output file: ')
	testing(test_doc,outputfile)
	print('file was generated')
