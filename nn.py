from layer import Layer
from df_utils import identity
from df_utils import softmax
from df_utils import sigmoid
from df_utils import relu
from df_utils import cross_entropy_loss
from df_utils import squared_loss
from train_utils import diag
from train_utils import eta
from train_utils import random_weights_gaussian

import numpy as np

from random import shuffle

class Neural_Net:
    
    def __init__(self, input_len, hidden_neurons, output_len, activation_functions,
                 loss_func, weight_init_func = np.zeros, output_func = identity):
        self.loss = loss_func
        self.output_func = output_func
        self.L = len(hidden_neurons) + 2
        self.layers = []
        
        input_layer = Layer(np.identity(input_len), np.zeros(input_len), identity, True, False)
        self.layers.append(input_layer)
        
        for neurons, func in zip(hidden_neurons, activation_functions):
            prev_neurons = self.layers[-1].num_neurons
            weight_matrix = weight_init_func((neurons, prev_neurons))
            biases = weight_init_func((neurons,))
            this_layer = Layer(weight_matrix, biases, func)
            self.layers.append(this_layer)
        
        prev_neurons = self.layers[-1].num_neurons
        out_layer_weight_matrix = weight_init_func((output_len, prev_neurons))
        biases = weight_init_func((output_len,))
        output_layer = Layer(out_layer_weight_matrix, biases, output_func, False, True)
        self.layers.append(output_layer)
        
    def __call__(self, x):
        return self.predict(x)
    
    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)[1]
        return output
    
    def train(self, train_data, val_data, test_data, step=2.0, epochs=100, learning_sched = eta):
        self.sgd(train_data, step, epochs, learning_sched, val_data, test_data)
    
    def avg_loss(self, data):
        avg_loss = 0.
        for (x,y) in data:
            prediction = self.predict(x)
            avg_loss += self.loss(prediction, y)
        avg_loss /= len(data)
        return avg_loss
    
    def check_acc(self, training_data, validation_data, test_data):
        correct_training = 0.
        correct_val = 0.
        correct_test = 0.
        for (x,y) in validation_data:
            prediction = np.argmax(self.predict(x))
            if y[prediction] == 1:
                correct_val += 1
        for (x,y) in training_data:
            prediction = np.argmax(self.predict(x))
            if y[prediction] == 1:
                correct_training += 1
        for (x,y) in test_data:
            prediction = np.argmax(self.predict(x))
            if y[prediction] == 1:
                correct_test += 1
        return correct_training / len(training_data), correct_val / len(validation_data), correct_test / len(test_data)
    
    def sgd(self, train_data, step, max_epochs, learning_sched, val_data, test_data):
        t = 0.
        initial_acc = self.check_acc(train_data, val_data, test_data)
        val_accs = []
        print "0 Initial training/validation accuracy:", initial_acc[0], initial_acc[1], initial_acc[2]
        val_accs.append(initial_acc[1])
        for epoch in range(1, max_epochs+1):
            shuffle(train_data)
            for x,y in train_data:
                self.update_weights(x, y, step * learning_sched(t))
                t += 1
            accuracies = self.check_acc(train_data, val_data, test_data)
            print epoch, "Training/validation/test accuracy:", \
            accuracies[0], accuracies[1], accuracies[2]
            val_accs.append(accuracies[1])
            print "Loss:", self.avg_loss(train_data)
            if len(val_accs) > 10 and val_accs[epoch - 10] >= val_accs[epoch]:
                print val_accs
                break
    
    def update_weights(self, x, y, step):
        dl_dW, dl_db = self.find_gradients(x, y)
        for i in range(1, self.L):
            layer = self.layers[i]
            layer.w -= step * dl_dW[i]
            layer.b -= step * dl_db[i]

    def find_gradients(self, x, y):
        L = self.L
        Z = []
        A = []
        err = [np.empty((1)) for _ in range(L)] # only err[1] through err[L-1] are meaningful; L-1 is output layer, 0 is input
        output = x
        for layer in self.layers:
            z, a = layer(output)
            Z.append(z)
            A.append(a)
            output = a
        err[L-1] = np.dot(self.output_func.derivative(Z[L-1]), self.loss.derivative(A[L-1], y))
        for i in range(L-2, 0, -1):
            layer = self.layers[i]
            err[i] = np.dot(self.layers[i+1].w.T, err[i+1])
            err[i] = np.dot(layer.a.derivative(Z[i]), err[i])
        dl_dW = [np.empty((1)) for _ in range(L)] # again ignoring dl_dW[0]
        dl_db = [np.empty((1)) for _ in range(L)] # biases
        for i in range(1, L):
            dl_db[i] = err[i]
            dl_dW[i] = np.dot(np.reshape(A[i-1], (A[i-1].shape[0],1)), np.reshape(err[i], (1,err[i].shape[0]))).T
            
        return dl_dW, dl_db
            