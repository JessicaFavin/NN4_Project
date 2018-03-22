import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata

import joblib # joblib version: 0.10.0 (dev)

from plotting import get_learning_curve, show_img


def create_imgs(num_img):
    generated_samples = []

    for gen in range(num_img):
        temp = []
        sample = np.random.randint(2, size=784)
        temp.append(sample)
        #print(len(temp))
        #print(temp)
        generated_samples.append(temp[0])
    
    generated_samples = np.matrix(generated_samples)

    return generated_samples


def set_data():

    ''' ######################### creation of dataset in another file ###############################################'''
    data = fetch_mldata('MNIST original')
    X = data['data']
    X = X / 255
    n = 60000
    X = X[:n]
    
    X = np.concatenate((X[::2], create_imgs(30000)), axis=0)
    
    ''' ######################### creation of labels in a file #####################################################'''
    obj1 = np.ones(30000)
    obj11= np.pad(obj1, (0, 30000), 'constant') #labels 1 if it's from MNIST, O if it's randomly generated
    y = np.asarray(obj11)
    
    joblib.dump(X, "part_3_X.pkl", compress=True)
    joblib.dump(y, "part_3_y.pkl", compress=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    joblib.dump(X_train, "part_3_X_train.pkl", compress=True)
    joblib.dump(X_test, "part_3_X_test.pkl", compress=True)
    joblib.dump(y_train, "part_3_y_train.pkl", compress=True)
    joblib.dump(y_test, "part_3_y_test.pkl", compress=True)
    
    return X, y

def set_mlp():
    X_train = joblib.load("part_3_X_train.pkl")
    y_train = joblib.load("part_3_y_train.pkl")
    
    #MPL Classifier
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,solver='adam', verbose=10, tol=1e-4, random_state=1,learning_rate ='adaptive')
    mlp.fit(X_train,y_train)
    
    joblib.dump(mlp, "part_3_mlp.pkl", compress=True)
    
    return mlp

def get_score():
    mlp = joblib.load("part_3_mlp.pkl")
    
    X_train = joblib.load("part_3_X_train.pkl")
    y_train = joblib.load("part_3_y_train.pkl")
    X_test = joblib.load("part_3_X_test.pkl")
    y_test = joblib.load("part_3_y_test.pkl")

    print("Training set score: %f" % mlp.score(X_train, y_train))

    print("Test set score: %f" % mlp.score(X_test, y_test))

if __name__ == '__main__':
    set_data()

    set_mlp()
    
    get_score()
    
    get_learning_curve(joblib.load("part_3_X.pkl"), joblib.load("part_3_y.pkl"))
    
