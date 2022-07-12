# Spam or Ham 📧

Spam or Ham is a program that detects whether an email is considered spam or "ham". The dataset for this project was provided from [UCI Machine Learning SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). This repository contains implementations of two different machine learning models a naive Bayes classifier and a neural network with the purpose of determining which model would be a better fit for the task. The nerual network is a simple perceptron model with a binary step activation function. From my implementations, I determined the naive Bayes classifier generalizes to the data set a lot better than the simple neural network, because in the [sample_results.txt](https://github.com/JonathanCen/Spam-or-Ham/blob/main/sample_results.txt) we can observe that when the neural network trains for 100 iterations it is able to achieve 100% accuracy on the training data, but doesn't perform as well on the test data in comparison to the naive bayes model.

## Getting Started

These instructions will give you a copy of the neural network up and running on
your local machine for development and testing purposes.

### Prerequisites

To run this application locally on your computer, you'll need `Git` and `python3` installed on your computer.

### Installing

Then run the following command in the command line and go to the desired directory to store this project:

Clone this repository:

    git clone https://github.com/JonathanCen/Spam-or-Ham.git

Navigate to the cloned repository:

    cd Spam-or-Ham

To run the naive bayes text classifier:

    python3 Naive_Bayes_Text_Classifier.py

To run the neural network text classifier:

    python3 Perceptron_Text_Classifier.py

## Quick notes about the results:

- Test_without_stopwords: running the test documents without the stop words through the perceptron that was trained with stop words
- Test_without_stopwords_on_new_perceptron: running the test documents without the stop words through a newly trained perceptron on documents without stop words
- Iterations/Learning Rates I choose:
  - Iterations = [5, 10, 15, 20, 100]
  - Learning Rates = [0.01, 0.05, 0.1, 0.5]

## Additional Notes:

Training:

- I initialized the weights of to small random numbers ranging between 0.0 to 1.0.
- Bias: I hard set the bias with a weight of 5 (did not update this value as I trained).
- Features: The features are the bag of words of that one document I am currently training on (not the entire training set).
- When I train my perceptron, I oscillate between training Ham & Spam documents (training: ham, spam, ham, spam, ...), and when I run out of spam docs, I randomly select spam documents within the training to continue training the model. I did this because there are unequal amount of documents, and I found that the weights of my perceptron was heavily dominated by the ham documents, so when I tested my model, it would correctly predict ham documents 100% of the time but for spam documents it will have an accuracy of 0%, so I found this as a great solution to solve the problem of the inadequate spam documents.
- Bias: I hard set the bias feature to 1 with a weight of 5.
- The activation function I used:
  - if feature_vector \* weight_vector > 0, then document is spam (1) else document is ham (-1)

Testing:

- Bias: Set to the same value as training
- Features: Same as the training (bag of words of the current document I am viewing)

## Contributing

All issues and feature requests are welcome.
Feel free to check the [issues page](https://github.com/JonathanCen/Spam-or-Ham/issues) if you want to contribute.

## Authors

- **Jonathan Cen** - [LinkedIn](https://www.linkedin.com/in/jonathancen/), [Github](https://github.com/JonathanCen)

## License

Copyright © 2022 [Jonathan Cen](<ADD PERSONAL WEBSITE LINK>).\
This project is [MIT licensed](https://github.com/JonathanCen/Spam-or-Ham/blob/main/LICENSE).
