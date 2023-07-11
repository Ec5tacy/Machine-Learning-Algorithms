# Machine-Learning-Algorithms

# Assignment 1
# Perceptron Learning Algorithm

## Model PM1

1. Load the dataset and split it into training and testing sets.
2. Initialize the weights of the perceptron to random values.
3. Iterate over the training set, and for each example, update the weights of the perceptron according to the perceptron learning rule.
4. Once all of the training examples have been processed, evaluate the performance of the perceptron on the testing set.

## Model PM2

1. Randomly shuffle the order of the training examples.
2. Re-train the perceptron on the shuffled training set.
3. Compare the performance of the new perceptron (PM2) to the original perceptron (PM1).

## Model PM3

1. Normalize the features of the dataset.
2. Re-train the perceptron on the normalized training set.
3. Compare the performance of the new perceptron (PM3) to the original perceptron (PM1).

## Model PM4

1. Randomly permute the order of the features in the dataset.
2. Build a classifier (Perceptron Model – PM4) on the permuted dataset.
3. Compare the performance of PM4 to PM1.

# Fisher's Linear Discriminant Analysis

## Model FLDM1

1. Calculate the mean and covariance matrices for the positive and negative classes.
2. Find the Fisher's linear discriminant function.
3. Use the Fisher's linear discriminant function to classify new examples.

## Model FLDM2

1. Randomly permute the order of the features in the dataset.
2. Build a Fisher's linear discriminant model (FLDM2) on the permuted training data.
3. Compare the performance of FLDM2 to FLDM1.

# Logistic Regression

## Model LR1

1. Fit a logistic regression model to the training data.
2. Evaluate the performance of the model on the testing data.

## Model LR2

1. Apply feature engineering to the training data.
2. Fit a logistic regression model to the modified training data.
3. Evaluate the performance of the model on the testing data.

# Comparative Study

1. Evaluate the performance of all of the models on the testing data.
2. Compare the performance of the models using a variety of metrics, such as accuracy, precision, recall, and F1 score.
3. Identify the best performing model.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Assignment 2


# Part A - Naive Bayes Classifier to predict income

## Data Preprocessing

1. Load the dataset into a pandas DataFrame.
2. Check for missing values and handle them appropriately.
3. Split the dataset into training and testing sets (80% for training, 20% for testing).

## Naive Bayes Classifier Implementation

1. Implement a function to calculate the prior probability of each class (benign and malignant) in the training set.
2. Implement a function to calculate the conditional probability of each feature given to each class in the training set.
3. Implement a function to predict the class of a given instance using the Naive Bayes algorithm.
4. Implement a function to calculate the accuracy of your Naive Bayes classifier on the testing set.

## Evaluation and Improvement

1. Evaluate the performance of your Naive Bayes classifier using accuracy, precision, recall, and F1-score.
2. Experiment with different smoothing techniques to improve the performance of your classifier.
3. Compare the performance of your Naive Bayes classifier with other classification algorithms like logistic regression and k-nearest neighbors.

# Part B: Building a Basic Neural Network for Image Classification

## Data

1. Load the MNIST dataset into a TensorFlow, Keras, or PyTorch dataset.

## Architecture

1. Build a basic neural network architecture that consists of an input layer, one or more hidden layers, and an output layer.
2. Vary one or more parameters from the following list:
    * Number of hidden layers – 2 or 3
    * Total number of neurons in the hidden layer is 100 or 150
    * Activation function is from any of the following functions: tanh, sigmoid, ReLu

## Training & Testing

1. Train your network on the MNIST dataset.
2. Use any optimization algorithm like stochastic gradient descent or Adam optimizer.
3. Evaluate your network's performance on a test set of images from the MNIST dataset.
4. Calculate the accuracy and confusion matrix to measure your network's performance.

## Comparative Study

1. Perform a comparative study of these 15 models and figure out the best classifier.
2. Do you have a classifier that is not statistically significant from the best classifier?
3. Detail the results with all explanations.
