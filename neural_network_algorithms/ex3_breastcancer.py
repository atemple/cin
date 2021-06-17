#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 21:06:56 2021

@author: atemple
"""
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class Perceptron(object):
    
    def __init__(self, learning_rate, epochs):
        self.epochs = epochs        
        self.learning_rate = learning_rate # learning rate
        self.bias = 0.0  # bias
        self.weights = None  # weights assigned to input features
        # count of errors during each iteration
        self.misclassified_samples = []

    def f(self, x: np.array):
        return np.dot(x, self.weights) + self.bias

    def predict(self, x: np.array):
        return np.where(self.f(x) >= 0, 1, -1)
    
    def fit(self, X, d, weight):
        self.weights = [weight for i in range(len(X[0]))]
        self.bias = 0.0
        self.misclassified_samples = []
        for _ in range(self.epochs):
            sum_error = 0
            for xi, di in zip(X, d):
                error = di - self.predict(xi)
                update = self.learning_rate * error
                self.weights += update * xi
                self.bias += update
                sum_error += error**2
            self.misclassified_samples.append(sum_error)
            print('>epoch=%d, lrate=%.3f, weight=%s, error=%.3f' % ((_+1), self.learning_rate, self.weights, sum_error))

                 
# load breast cancer dataset
cancer = load_breast_cancer()

# get breast cancer data (X) and target (d)
data_set = cancer.data
data_target = cancer.target
data_features = cancer.feature_names

print(data_features)
print(data_target)
print(data_set)

# split the data
x_train, x_test, d_train, d_test = train_test_split(data_set, data_target, test_size=0.15)

learning_rate=0.1
weight=1
epochs = 50
classifier = Perceptron(learning_rate, epochs)
classifier.fit(x_train, d_train, weight)

# plot the number of errors during each iteration
plt.plot(range(1, classifier.epochs + 1),
         classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()

# run predict classifier
x_predict = classifier.predict(x_test)

misclassified = 0
for predict, verify in zip(x_predict, d_test):
    if predict != verify:
        misclassified += 1
print("\nTest size: " + str(len(x_predict)) + "\n")
print("The number of misclassified: " + str(misclassified) + "\n")

print("Accuracy score: %f\n" % accuracy_score(x_predict, d_test))
print("\nConfusion matrix: \n", confusion_matrix(x_predict, d_test))
print("\nClassification report: \n", classification_report(x_predict, d_test))

