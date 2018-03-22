#!/usr/bin/python
import time
import matplotlib.pyplot as plt
import numpy as np

from random import randint
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

#Part 2

#Question 1

mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target
print(len(X))
plt.matshow(X[20000].reshape(28,28))
plt.gray()
plt.show()
print("startmlp")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20000)
NN = MLPClassifier(hidden_layer_sizes = (30,30,30), solver = 'adam', learning_rate_init = 10**(-3))
print("mlp done")
NN.fit(X_train, y_train)
print("fit done")
pred = NN.predict(X_test)
print("pred done")
print(NN.score(X_test, y_test, sample_weight=None))

i = randint(0, X_test.shape[0])
plt.matshow(X_test[i].reshape(28,28))
plt.gray()
plt.show()
print("predict: {}, label: {}".format(NN.predict(X_test[i].reshape(1, -1)), y_test[i]))
cm = confusion_matrix(y_test, pred)
print(cm)

#Question 2 : cross validation

#Question 3 : Probabilities estimation

#Question 4 : Probabilities estimation with NN

cr = classification_report(y_test, pred)
print(cr)

#Question 5 : Classification and rejection
