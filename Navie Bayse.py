import nltk
import parse
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from parse import parse
from nltk.tokenize import RegexpTokenizer
file1 = open("corpus1_train.labels")
label_dict = {} # the prob of each label
map_label_map_word = {} # the prob of each word for each label
prob_dict = {} # the prob of each label
number_word_label = {} # number of words for each label
number_training_files = 0
number_training_words = 0
number_distinct_words = {}
number_totalwords_perlabel = {}
input_words = []

stop_words = set(stopwords.words('english'))
#print(stop_words)

for line in file1:
    number_training_files += 1.0
    path,label = parse.parse("{} {}", line).fixed
    if label not in label_dict:
        label_dict[label] = 1
        map_label_map_word[label] = {}
    else:
        label_dict[label] = label_dict[label]+1

    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    words = [stemmer.stem(w) for w in tokenizer.tokenize(open(path, 'r').read().lower())]
    words = [word for word in words if word not in stop_words] # clean the words

    for e in words:
        if e not in map_label_map_word[label]:
            map_label_map_word[label][e] = 1
        else:
            map_label_map_word[label][e] = map_label_map_word[label][e] + 1

file1.close()

for label in label_dict:
    label_dict[label] = label_dict[label]/number_training_files

for label in map_label_map_word.keys():
    number_distinct_words[label] = len(map_label_map_word[label])
    number_totalwords_perlabel[label] = sum(map_label_map_word[label].values())

print(label_dict) # the percent of each label
#print(map_label_map_word)
print(number_totalwords_perlabel)
print(number_distinct_words)

def parsing(path):
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer('r\w+')
    words = [stemmer.stem(w) for w in tokenizer.tokenize(open(path, 'r').read().lower())]
    return words
def reset_dict():
    for label in prob_dict:
        prob_dict[label] = 0;
def count_percent(words,label):
    total_count = 0
    number_unknwn = 0;
    for e in words:
        if e in stop_words:
            continue
        if e not in map_label_map_word[label]:
            number_unknwn += 1
    for e in words:
        if e in stop_words:
            continue
        if e not in map_label_map_word[label]:
            total_count += 1 / (number_distinct_words[label]*number_unknwn + number_totalwords_perlabel[label])
        else:
            total_count += (map_label_map_word[label][e] + number_unknwn) / (number_totalwords_perlabel[label] + number_unknwn*number_distinct_words[label])
    percent = label_dict[label] * total_count
    #print(percent)
    #print(label)
    return percent
def findmax():
    res = ""
    max = -10000000
    for label in prob_dict:
        if prob_dict[label] > max:
            res = label
            max = prob_dict[label]
    return res
test_file = open("corpus1_test.list")
output_file = input("enter the name of the output file: ")
output_file = open(output_file,"w")
for line in test_file:
    path = parse.parse("{}",line).fixed
    test_doc = open(path[0])
    words = parsing(path[0])
    for label_element in label_dict.keys():
        percent = count_percent(words,label_element)
        prob_dict[label_element] = percent
    #print(prob_dict) print the prob of four labels
    result = findmax() # find the max prob of four labels
    output_file.write(path[0] + ' ')
    output_file.write(result + '\n')
    reset_dict()
    test_doc.close()
output_file.close()
