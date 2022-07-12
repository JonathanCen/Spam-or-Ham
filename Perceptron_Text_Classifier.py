import numpy as np
import os
import sys
import string
import re
from nltk.stem.snowball import SnowballStemmer
import random
import copy
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
init_feature_weights = {}
new_feature_weights = {}
train_ham_list = []
train_spam_list = []
r_train_ham_list = []
r_train_spam_list = []

# If document are spams, then target_val = 1; else target_val = -1
def test_perceptron(documents, target_val):
    global new_feature_weights
    num_correct = 0
    bias_weight = 5     # Bias feature = 1

    # Run through all the document nad increment the num_correct counter
    for doc in documents:

        # Feature vector: bag of word of the current document
        doc_bag_of_words = {}
        for word in doc:
            if word in doc_bag_of_words:
                doc_bag_of_words[word] += 1
            else:
                doc_bag_of_words[word] = 1

        # Compute the linear combination of the feature vector and weight vector
        running_linear_combination = 0
        for word in doc:
            if word in new_feature_weights:
                running_linear_combination += (new_feature_weights[word] * doc_bag_of_words[word])
        running_linear_combination += bias_weight

        # Make a prediction
        prediction = 1 if running_linear_combination > 0 else -1
        if prediction == target_val:
            num_correct += 1

    return num_correct


# When training it will be oscillating between HAM and SPAM documents.
# target_val = 1 if documets == spam
# target_val = -1 if documents == ham
def training_perceptron(iterations, learning_rate, switch = 0):
    global init_feature_weights, train_spam_list, train_ham_list, r_train_ham_list, r_train_spam_list, bias_w
    new_feature_weights = copy.deepcopy(init_feature_weights)       # create a new copy of the feature weights for this NN
    ham_list = r_train_ham_list if switch else train_ham_list       # selecting different lists depending if we training with stop words or not
    spam_list = r_train_spam_list if switch else train_spam_list
    bias_weight = 5                                                 # bias feature = 1

    for x in range(iterations):
        # We know that there are more HAM documents so when we run out of SPAM documents to oscillate it will randomly select documents
        # Update weights using HAM then SPAM
        for num, ham_doc in enumerate(ham_list):
            # ----------------- Update using HAM (-1) -----------------
            # Feature vector: bag of word of the current document
            doc_bag_of_words = {}
            for word in ham_doc:
                if word in doc_bag_of_words:
                    doc_bag_of_words[word] += 1
                else:
                    doc_bag_of_words[word] = 1

            # Compute the linear combination of all the weights and the words
            running_linear_combination = 0
            for word in ham_doc:
                # this check is most likely useless...
                if word in new_feature_weights:
                    running_linear_combination += (new_feature_weights[word] * doc_bag_of_words[word])
            running_linear_combination += bias_weight

            # Make a prediction
            prediction = 1 if running_linear_combination > 0 else -1

            # Update the weights based on words in the document
            for word in ham_doc:
                if word in new_feature_weights:
                    delta_weight = learning_rate * (-1 - prediction) * doc_bag_of_words[word]
                    new_feature_weights[word] = new_feature_weights[word] + delta_weight # hopefully this does it properly


            # ----------------- Update using SPAM (1) -----------------
            # Either randomly choose a document (only happens if we run of documents) or choosing the corresponding document
            spam_doc = None
            if (num >= len(spam_list)):
                spam_doc = spam_list[random.randint(0, len(spam_list)-1)]
            else:
                spam_doc = spam_list[num]

            # Feature vector: bag of word of the current document
            doc_bag_of_words = {}
            for word in spam_doc:
                if word in doc_bag_of_words:
                    doc_bag_of_words[word] += 1
                else:
                    doc_bag_of_words[word] = 1
            
            # Compute the linear combination of all the weights and the words
            running_linear_combination = 0
            for word in spam_doc:
                if word in new_feature_weights:
                    running_linear_combination += (new_feature_weights[word] * doc_bag_of_words[word])
            running_linear_combination += bias_weight

            # Make a prediction
            prediction = 1 if running_linear_combination > 0 else -1

            # Update the weights based on words in the document
            for word in spam_doc:
                if word in new_feature_weights:
                    delta_weight = learning_rate * (1 - prediction) * doc_bag_of_words[word]
                    new_feature_weights[word] = new_feature_weights[word] + delta_weight # hopefully this does it properly

    return new_feature_weights


# generate random corresponding weights for each unique word
def generate_weights(bag_of_words):
    feature_weights = {}
    for key, values in bag_of_words.items():
        feature_weights[key] = random.random()
    return feature_weights


# generates features/bag of words from both data files
def build_bag_of_words(data_spam, data_ham):
    bag_of_words = {}
    for list_words in data_spam:
        for words in list_words:
            if words in bag_of_words:
                bag_of_words[words] = 1 + bag_of_words[words]
            else:
                bag_of_words[words] = 1
    for list_words in data_ham:
        for words in list_words:
            if words in bag_of_words:
                bag_of_words[words] = 1 + bag_of_words[words]
            else:
                bag_of_words[words] = 1
    return bag_of_words


# extracts words from docs
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


# removes stop words from docs
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
    

## TASK: Report 20 suitable combinations of number of iterations and the learning rate
def main():

    # This were results from my previous HW
    # I've attach the python file within this zip file that outputs these results
    print("Naive Bayes Results:")
    print("{:<25}{:^10}{:>25}".format("Training", "Test_with_stopwords", "Test_without_stopwords"))
    print("{:<25}{:^10}{:>25}\n".format("0.9913606911447084", "0.9602510460251046", "0.9581589958158996"))

    """
    Possible optimizations/changes:
        - Different hard set bias values
        - Updating bias (changing the bias weight when updating the weights of the perceptron)
        - Features (bag of words of the entire training file or bag of words of the current document)
    """
    training_folder = "train"
    test_folder = "test"

    global init_feature_weights, new_feature_weights, train_ham_list, train_spam_list, r_train_ham_list, r_train_spam_list

    ## Extract the files from the directory
    train_ham_files = os.listdir(os.path.join(os.path.join(os.getcwd(), training_folder), 'ham'))
    train_spam_files = os.listdir(os.path.join(os.path.join(os.getcwd(), training_folder), 'spam'))
    test_ham_files = os.listdir(os.path.join(os.path.join(os.getcwd(), test_folder), 'ham'))
    test_spam_files = os.listdir(os.path.join(os.path.join(os.getcwd(), test_folder), 'spam'))

    ## extract the words from the files
    train_ham_list = preprocessing(train_ham_files, os.path.join(os.path.join(os.getcwd(), training_folder), 'ham'))
    train_spam_list = preprocessing(train_spam_files, os.path.join(os.path.join(os.getcwd(), training_folder), 'spam'))
    test_ham_list = preprocessing(test_ham_files, os.path.join(os.path.join(os.getcwd(), test_folder), 'ham'))
    test_spam_list = preprocessing(test_spam_files, os.path.join(os.path.join(os.getcwd(), test_folder), 'spam'))

    ## remove stop words from train & test documents
    r_train_ham_list = remove_stop_words_list(train_ham_list)
    r_train_spam_list = remove_stop_words_list(train_spam_list)
    r_test_ham_list = remove_stop_words_list(test_ham_list)
    r_test_spam_list = remove_stop_words_list(test_spam_list)

    # generate the features and weights
    bag_of_words = build_bag_of_words(train_ham_list, train_spam_list)
    init_feature_weights = generate_weights(bag_of_words)

    # training iterations/learning rates
    iterations = [5, 10, 15, 20, 100]
    learning_rates = [0.01, 0.05, 0.1, 0.5]

    # run through all iterations then learning_rates individually
    print("\nPerceptron Results:")
    for iteration in iterations:
        for l_r in learning_rates:
            # train NN w/ stop words
            new_feature_weights = training_perceptron(iteration, l_r)
            train_results = (test_perceptron(train_spam_list, 1) + test_perceptron(train_ham_list, -1)) / (len(train_spam_list) + len(train_ham_list))
            test_results = (test_perceptron(test_spam_list, 1) + test_perceptron(test_ham_list, -1)) / (len(test_spam_list) + len(test_ham_list))

            # running docs w/o stop words on perceptron trained with stop word
            test_without_stop_words = (test_perceptron(r_test_spam_list, 1) + test_perceptron(r_test_ham_list, -1)) / (len(r_test_spam_list) + len(r_test_ham_list))

            # train a new perceptron without the stop words in the training data
            new_feature_weights = training_perceptron(iteration, l_r, 1)
            test_without_stop_words_new_NN = (test_perceptron(r_test_spam_list, 1) + test_perceptron(r_test_ham_list, -1)) / (len(r_test_spam_list) + len(r_test_ham_list))

            print(f"W/ iterations: {iteration} and learning_rate: {l_r}:")
            print("{:<25}{:^10}{:>25}{:>45}".format("Training", "Test_with_stopwords", "Test_without_stopwords", "Test_without_stopwords_on_new_perceptron"))
            print("{:<25}{:^10}{:>25}{:>35}\n".format(train_results, test_results, test_without_stop_words, test_without_stop_words_new_NN))


if __name__ == "__main__":
    main()