import numpy as np
import os
import sys
import string
import re
from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.porter import *  

# stop words
stop_words = set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 
'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 
'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', 
"don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 
'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 
'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 
'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 
'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 
"shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 
'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", 
"they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 
'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 
'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", 
"you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])

# global variables
spam_bag_of_words = {}
spam_N = 0              # Laplace Smoothing's N
num_spam_files = 0      # Naive's prior
ham_bag_of_words = {}
ham_N = 0               # Laplace Smoothing's N
num_ham_files = 0       # Naive's prior


# calculates laplace_estimate for both spam and ham
# returns the log of the calculations
def laplace_estimate(words, X, k=1):
    global spam_bag_of_words, ham_bag_of_words, spam_N, ham_N

    # return estimates
    spam_lapace_est = 0
    ham_lapace_est = 0 
    # loops through all the stemmed words in the file
    for word in words:
        spam_lapace_est += np.log((((spam_bag_of_words[word] if word in spam_bag_of_words else 0) + 1) / (spam_N + X)))
        ham_lapace_est += np.log((((ham_bag_of_words[word] if word in ham_bag_of_words else 0) + 1) / (ham_N + X)))
    return [spam_lapace_est, ham_lapace_est]


def naive_bayes(list_document_words, file_type):

    global spam_bag_of_words, ham_bag_of_words, num_spam_files, num_ham_files

    num_correct = 0
    num_files = len(list_document_words)
    X = len(set(list(spam_bag_of_words.keys()) + list(ham_bag_of_words.keys())))

    for document_words in list_document_words:
        spam_lapace_est, ham_lapace_est = laplace_estimate(document_words, X, 1)

        spam_lapace_est += np.log(num_spam_files / (num_spam_files + num_ham_files))
        ham_lapace_est += np.log(num_ham_files / (num_spam_files + num_ham_files))

        if (spam_lapace_est > ham_lapace_est and file_type == "spam"): 
            num_correct += 1
        elif (spam_lapace_est <= ham_lapace_est and file_type == "ham"): 
            num_correct += 1

    return num_correct

def build_bag_of_words(data):
    bag_of_words = {}
    for list_words in data:
        for words in list_words:
            if words in bag_of_words:
                bag_of_words[words] = 1 + bag_of_words[words]
            else:
                bag_of_words[words] = 1
    return bag_of_words


def preprocessing(files, path):
    data = []
    #stemmer = PorterStemmer()
    stemmer = SnowballStemmer("english")
    for file in files:
        f = open(os.path.join(path, file), 'r', errors='ignore')
        raw = f.read()
        f.close()
        filtered = re.sub(r'[^\w\s]+', '', raw).split()
        stemmed = [stemmer.stem(word) for word in list(filtered)]
        data.append(stemmed)
    return data

def remove_stop_words_list(documents):
    global stop_words
    new_documents = []
    for doc in documents:
        temp_doc = []
        for word in doc:
            if word not in stop_words:
                temp_doc.append(word)
        new_documents.append(temp_doc)
    return new_documents
    

def main():

    ## loops through the directory and builds the bag of words
    training_folder = "train"
    test_folder = "test"

    train_ham_files = os.listdir(os.path.join(os.path.join(os.getcwd(), training_folder), 'ham'))
    train_spam_files = os.listdir(os.path.join(os.path.join(os.getcwd(), training_folder), 'spam'))
    test_ham_files = os.listdir(os.path.join(os.path.join(os.getcwd(), test_folder), 'ham'))
    test_spam_files = os.listdir(os.path.join(os.path.join(os.getcwd(), test_folder), 'spam'))

    train_ham_list = preprocessing(train_ham_files, os.path.join(os.path.join(os.getcwd(), training_folder), 'ham'))
    train_spam_list = preprocessing(train_spam_files, os.path.join(os.path.join(os.getcwd(), training_folder), 'spam'))
    test_ham_list = preprocessing(test_ham_files, os.path.join(os.path.join(os.getcwd(), test_folder), 'ham'))
    test_spam_list = preprocessing(test_spam_files, os.path.join(os.path.join(os.getcwd(), test_folder), 'spam'))

    global spam_bag_of_words, ham_bag_of_words, spam_N, ham_N, num_ham_files, num_spam_files

    num_ham_files = len(train_ham_files)
    num_spam_files = len(train_spam_files)

    ham_N = 0
    for list_words in train_ham_list:
        ham_N += len(list_words)
    spam_N = 0
    for list_words in train_spam_list:
        spam_N += len(list_words)

    ham_bag_of_words = build_bag_of_words(train_ham_list)
    spam_bag_of_words = build_bag_of_words(train_spam_list)

    ## Part 1
    # test training
    training_estimate = (naive_bayes(train_ham_list, "ham") + naive_bayes(train_spam_list, "spam")) / (num_ham_files + num_spam_files)

    # test folder
    test_estimate = (naive_bayes(test_ham_list, "ham") + naive_bayes(test_spam_list, "spam")) / (len(test_ham_files) + len(test_spam_files))

    ## Part 2
    # remove stop words in training docs
    train_ham_list = remove_stop_words_list(train_ham_list)
    train_spam_list = remove_stop_words_list(train_spam_list)

    ham_N = 0
    for list_words in train_ham_list:
        ham_N += len(list_words)

    spam_N = 0
    for list_words in train_spam_list:
        spam_N += len(list_words)

    # build_bag_of_words
    ham_bag_of_words = build_bag_of_words(train_ham_list)
    spam_bag_of_words = build_bag_of_words(train_spam_list) 

    # remove stop words in test docs
    new_test_ham_list = remove_stop_words_list(test_ham_list)
    new_test_spam_list = remove_stop_words_list(test_spam_list)

    r_test_estimate = (naive_bayes(new_test_ham_list, "ham") + naive_bayes(new_test_spam_list, "spam")) / (len(new_test_ham_list) + len(new_test_spam_list))

    print("{:<25}{:^10}{:>25}".format("Training", "Test_with_stopwords", "Test_without_stopwords"))
    print("{:<25}{:^10}{:>25}".format(training_estimate, test_estimate, r_test_estimate))


if __name__ == "__main__":
    main()