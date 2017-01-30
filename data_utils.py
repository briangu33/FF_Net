import numpy as np

from numpy import loadtxt

training_size = 200
val_size = 150
test_size = 150

def get_single_digit(digit):
    print "getting data for " + str(digit)
    path = 'data/mnist_digit_' + str(digit) + '.csv'
    data = loadtxt(path)[:training_size + val_size + test_size]
    data = map(lambda x: map(lambda y: 2*(float(y)/255)-1, x), data) #normalize data
    # data = map(lambda x: map(lambda y: float(y), x), data)
    X_train = data[:training_size]
    X_val = data[training_size:training_size + val_size]
    X_test = data[training_size + val_size:]
    return X_train, X_val, X_test

def get_classification_data():
    X_train = []
    X_val = []
    X_test = []
    Y_train = []
    Y_val = []
    Y_test = []

    for digit in range(10):
        data = get_single_digit(digit)
        X_train += data[0]
        X_val += data[1]
        X_test += data[2]
        Y_train += [[0 if i != digit else 1 for i in range(10)] for _ in range(training_size)]
        Y_val += [[0 if i != digit else 1 for i in range(10)] for _ in range(val_size)]
        Y_test += [[0 if i != digit else 1 for i in range(10)] for _ in range(test_size)]

    return np.array(X_train), np.array(X_val), np.array(X_test), \
        np.array(Y_train), np.array(Y_val), np.array(Y_test)