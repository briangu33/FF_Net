# Feedforward Neural Net
A fully-connected vanilla feedforward neural network written in Python which can be used for classification problems, including MNIST handwritten digit recognition. Written for MIT 6.867.
Allows for the following modifications:
 - Variable number of layers and neurons in each layer
 - Different loss functions (cross entropy, squared...)
 - Choice of activation functions at every layer, including output layer (sigmoid, ReLU, tanh, identity, softmax...)
 - Freedom over initialization and learning schedule strategies
Achieves 90%+ accuracy on MNIST dataset (10-class digit classification) with 2 hidden layers (100 neurons per layer). Graphs below show accuracy on test dataset against training epochs (truncated at 100 epochs).
![val10.png](https://www.dropbox.com/s/tlw3vly01pdabp0/val10.png?dl=0&raw=1)
![val100.png](https://www.dropbox.com/s/9rkradbyaf6k39k/val100.png?dl=0&raw=1)
![val500.png](https://www.dropbox.com/s/ps3z3g3adueodyk/val500.png?dl=0&raw=1)
![val1010.png](https://www.dropbox.com/s/9xs6xlu2dhk8xty/val1010.png?dl=0&raw=1)
![val100100.png](https://www.dropbox.com/s/0ql76tx3l8ag9vw/val100100.png?dl=0&raw=1)
