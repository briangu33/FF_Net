from nn import Neural_Net

from data_utils import get_classification_data
from df_utils import cross_entropy_loss
from df_utils import squared_loss
from df_utils import identity
from df_utils import sigmoid
from df_utils import softmax
from df_utils import relu
from train_utils import random_weights_gaussian


X_train, X_val, X_test, Y_train, Y_val, Y_test = get_classification_data()

nn_mnist = Neural_Net(784, [100,100], 10, [relu, relu], cross_entropy_loss, random_weights_gaussian, softmax)
train_data = zip(X_train, Y_train)
val_data = zip(X_val, Y_val)
test_data = zip(X_test, Y_test)
nn_mnist.train(train_data, val_data, test_data)
