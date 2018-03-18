#!/usr/bin/python
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.neural_network import MLPClassifier

#train_samples = 5000

mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target
print(len(X))
plt.matshow(X[20000].reshape(28,28))
plt.gray()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000)
NN = MLPClassifier(
hidden_layer_sizes = (9,9),
activation = 'relu',
solver = 'lbfgs',
learning_rate_init = 10**(-5))

'''digits=load_digits()

NN = MLPClassifier(
hidden_layer_sizes = (9,9),
activation = 'relu',
solver = 'lbfgs',
learning_rate_init = 10**(-5))

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20)
NN.fit(X_train, y_train)
print(NN.score(X_test, y_test, sample_weight=None))'''
