# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:43:58 2018

@author: Célande
"""

from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

from plotting import plot_confusion_matrix, show_img, plot_classification_report

"""
    1 image is grayscale
    28x28
    with value 0 -> 255

    X_train, y_train <=> 60 000 first data
    X_test, y_test <=> 10 000 data left
"""

# set the seed
np.random.seed(42)

def percent_pixel_dataset():
    X = fetch_mldata ("MNIST original").data
    percent = np.zeros( 256 )
    for image in X:
        for pixel in image:
            percent[int(pixel)] += 1

    for i in range(len(percent)):
        percent[i] = int((percent[i] / (len(X) * 256)) * 100000)
    #print("percent = {}".format(percent))
    #choices = [[i]*percent[i] for i in range(len(percent))]
    #percent = joblib.load("percent_pixel.pkl")
    s = [sum([[i]*int(percent[i])], []) for i in range(len(percent))]
    choices = []
    for a in s:
        for b in a:
            choices.append(b)
    #print("choices = {}".format(choices))
    #print("shape = {}".format(np.shape(choices)))
    joblib.dump(choices, "part_1_percent_pixel.pkl", compress=True)

# change a grayscale 1D image
def data_format(data, nb_pixel):
    """
    for i in range(0, data.size):
        if data[i] > 0:
            data[i] = 255
    """

    for i in range(0, data.size):
        if data[i] < 5:
            data[i] = 0
    return data

# change image dataset into binary image dataset
def set_format(data, nb_pixel):
    result = []
    for i in range(0, data.shape[0]):
        result.append(data_format(data[i], nb_pixel))

    return np.array(result)

def data_formating(X_train, X_test):
    print("Setting train format")
    X_train = set_format(X_train, nb_pixel)
    print("Setting test format")
    X_test = set_format(X_test, nb_pixel)

    return X_train, X_test

def data_preprocessing(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    print("Preprocessing training set")
    X_train = scaler.transform(X_train)
    print("Preprocessing testing set")
    X_test = scaler.transform(X_test)

    return X_train, X_test

# load dataset and save into pkl files
def set_dataset(test_size, nb_pixel):
    mns = fetch_mldata ("MNIST original")
    target_names = np.unique(mns.target)

    X = mns.data
    y = mns.target

    joblib.dump(target_names, "part_1_target_names.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train, X_test = data_formating(X_train, X_test)
    X_train, X_test = data_preprocessing(X_train, X_test)

    joblib.dump(X_train, "part_1_X_train.pkl", compress=True)
    joblib.dump(y_train, "part_1_y_train.pkl", compress=True)

    joblib.dump(X_test, "part_1_X_test.pkl", compress=True)
    joblib.dump(y_test, "part_1_y_test.pkl", compress=True)

    print("Format set")

# load training dataset and save classifer into pkl file
def set_classifier(iteration, *nb_neurons):

    X_train = joblib.load("part_1_X_train.pkl")
    y_train = joblib.load("part_1_y_train.pkl")

    print("Setting classifier with {} neurons".format(nb_neurons))
    classifier = MLPClassifier(hidden_layer_sizes=(nb_neurons), solver='adam', learning_rate_init = iteration, random_state=42)
    print("Fitting classifier")
    classifier.fit(X_train, y_train)
    print("Classifier done")

    joblib.dump(classifier, "part_1_classifier.pkl", compress=True)

    print("Setting predictions")
    X_test = joblib.load("part_1_X_test.pkl")
    predictions = classifier.predict(X_test)
    joblib.dump(predictions, "part_1_predictions.pkl", compress=True)

    return classifier

# load testing dataset and classifier and score
def score_classifier(classifier=None):
    X_test = joblib.load("part_1_X_test.pkl")
    y_test = joblib.load("part_1_y_test.pkl")

    if classifier == None:
        classifier = joblib.load("part_1_classifier.pkl")

    print("Scoring classifier")
    print(classifier.score(X_test, y_test))

def predict_img(img, label, nb_pixel, classifier=None):
    show_img(img, nb_pixel)
    if classifier == None:
        classifier = joblib.load("part_1_classifier.pkl")
    print("predict: {}, label: {}".format(classifier.predict(img.reshape(1, -1)), label))

def test_predict(nb_pixel, classifier=None):
    X_test = joblib.load("part_1_X_test.pkl")
    y_test = joblib.load("part_1_y_test.pkl")
    i = randint(0, X_test.shape[0])
    #print("X_test i = {}".format(X_test[i]))
    predict_img(X_test[i], y_test[i], nb_pixel, classifier)

def init_cls(test_size, nb_pixel, iteration, set_data=True, *nb_neurons):
    if set_data:
        set_dataset(test_size, nb_pixel)
    return set_classifier(iteration, *nb_neurons)

def display_confusion_matrix(normalized=False, predictions=None):
    if predictions == None:
        predictions = joblib.load("part_1_predictions.pkl")

    y_test = joblib.load("part_1_y_test.pkl")

    """
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    """

    cm = confusion_matrix(y_test, predictions)

    class_names = joblib.load("part_1_target_names.pkl")

    if normalized:
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names, normalize=True,
                              title='Confusion matrix')
    else:
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names,
                              title='Confusion matrix')

def display_classification_report(predictions=None):
    if predictions == None:
        predictions = joblib.load("part_1_predictions.pkl")
    y_test = joblib.load("part_1_y_test.pkl")

    """
    print("Classification report:")
    print(classification_report(y_test, predictions))
    """
    cr = classification_report(y_test, predictions)
    plot_classification_report(cr)

def get_cross_val():
    from sklearn import svm
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score
    from sklearn import metrics

    X_train = joblib.load("part_1_X_train.pkl")
    y_train = joblib.load("part_1_y_train.pkl")
    X_test = joblib.load("part_1_X_test.pkl")
    y_test = joblib.load("part_1_y_test.pkl")
    mnist = fetch_mldata ("MNIST original")

    print("Setting svm classifier")
    clf = svm.SVC(kernel='linear', C=1, random_state=0).fit(X_train, y_train)
    joblib.dump(clf, "part_1_svm_clf.pkl", compress=True)

    print("Scoring classifier")
    print("score = {}".format(clf.score(X_test, y_test)))

    print("Setting cv")
    #n_samples = iris.data.shape[0]
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    joblib.dump(cv, "part_1_cv.pkl", compress=True)
    #print("Setting Cross val score")
    #print("score = {}".format(cross_val_score(clf, mnist.data, mnist.target, cv=cv)))

    print("Setting Cross Validation")
    scores = cross_validate(clf, mnist.data, mnist.target, scoring='precision_macro', return_train_score=False)
    print("scores keys = {}".format(sorted(scores.keys())))
    predicted = cross_val_predict(clf, mnist.data, mnist.target, cv=10)
    print("accuracy score = {}".format(metrics.accuracy_score(mnist.target, predicted)))

if __name__ == '__main__':
    nb_pixel = 28
    nb_neurons = [50, 20, 20]
    test_size = 0.40
    iteration = 0.001

    """
    # 10, 8, 6: 91%
    # 10, 8, 6, 4, 2: 86%
    # 12, 8, 6: 92%
    # 20, 8, 6: 92.9%
    # 20, 15, 6: 93.5%
    # 20, 20, 6: 93.9%
    # 20, 20, 20: 94%
    # 50, 20, 20: 96%
    # 50, 50, 50: 96.1%
    # 100, 50, 50: 96.4%
    # 100, 100, 100, 100: 96.7%
    # 100, 100, 100: 96.8%
    # 100, 100: 96.8%
    # 500, 500: 97%
    # 500, 500, 500: 97.2% # take a lot of time
    # 550, 500: 96.9%
    # 500, 400: 97.2% # take a lot of time
    # 500, 300: 96.98%
    # 500, 450: 97.19% # take a lot of time
    # 450, 450: 96.28% # take a lot of time
    # 500, 400, 400: 97.5% # take a lot of time
    # 500, 400, 500: 97.3%# take a lot of time
    # 500, 400, 300: 97.3%
    """
    #init_cls(test_size, nb_pixel, iteration, True, *nb_neurons)

    #score_classifier()

    #test_predict(nb_pixel)

    display_confusion_matrix()
    display_classification_report()

    #percent_pixel_dataset()

    #get_cross_val()
