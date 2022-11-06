import numpy as np
from neural_network import NeuralNetwork

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x) :
    return sigmoid(x)*(1-sigmoid(x))

def binary_cross_entropy(y_truth, y_pred):
    #For single values
    y_pred = np.clip(y_pred, 1e-7, 1- 1e-7)
    return y_truth* np.log(y_pred) + (1 - y_truth) * np.log(1-y_pred)

class PerceptronNetwork(NeuralNetwork):
    def __init__(self, weight_path=None, num_neuron=784, lr=0.01) -> None:
        super().__init__(weight_path, num_neuron, lr)

        self.bias = np.random.uniform(low=0, high=0.5, size= (1,))
        self.f = lambda x : 1 if x > 0 else 0

    def forward(self, input):
        output = sum(input * self.weights) + self.bias
        return self.f(output)    

    def update(self, input, output, ground_truth = None):
        self.weights += self.lr * (ground_truth - output) * input
        self.bias    += self.lr * (ground_truth - output)

    # def backprop_linear(self, input, output_error):
    #     input_error = sum(output_error * self.weights)
    #     weights_gradient = sum(input * output_error)
    #     bias_gradient = output_error

    #     self.weights -= weights_gradient * self.lr
    #     self.bias    -= bias_gradient * self.lr 

    #     return input_error        