import numpy as np
import os
from network.neural_network import NeuralNetwork
from utils.activation import *
np.set_printoptions(suppress=True)

class FeedForwardNN(NeuralNetwork):
    def __init__(self, weight_path=None, layer_neurons=[200], lr=0.01, input_size=784, output_size=10, 
        activation_function=None, 
        output_activation = 'sigmoid',
        H = 0.75, L =0.25,
        momentum = 0.5,
        loss_function = 'mse',
        ):
        super().__init__(weight_path, layer_neurons, lr, input_size, output_size, activation_function, output_activation, momentum=momentum, loss_function=loss_function)
        self.H, self.L = H,L
        self.input = []
        self.layer_types = []

        for i in range(len(self.weights)):
            self.layer_types.append('fc')
            if i == (len(self.weights) - 1):
                self.layer_types.append('output_act')            
            else:
                self.layer_types.append('hidden_act')

    def thresholding(self, x):
        new_x = []
        for b in range(x.shape[0]):
            minibatch_output = []
            for e in range(x.shape[1]):
                element = x[b][e]
                if element >= self.H: minibatch_output.append(1)
                elif element <= self.L: minibatch_output.append(0)
                else: minibatch_output.append(element)
            new_x.append(minibatch_output)

        return np.array(new_x)

    def forward(self, x):
        for i in range(len(self.weights)):
            self.input.append(x)  
            # print(x.shape, self.weights[i].shape, self.biases[i].shape)
            x = np.dot(x, self.weights[i]) + self.biases[i]
            self.input.append(x)

            if i == (len(self.weights) - 1):
                x = self.out_act(x)
                # print(x, x.shape)
                threshold_x = self.thresholding(x) 
            else:
                x = self.f(x)
        
        return x, threshold_x
    
    def clear_input(self):
        self.input = []

    def backward(self, loss):
        layer_output_error = loss
        weight_index = -1
        for i in range(-1, -len(self.layer_types) -1, -1):
            # print(i, self.layer_types[i])
            layer_output_error = self.layer_backprop(
                layer_input=self.input[i],
                layer_output_error=layer_output_error,
                weight_index=weight_index,
                layer_type=self.layer_types[i]
            )
            if self.layer_types[i] == 'fc':
                weight_index -= 1

        self.input = []

if __name__ == '__main__':
    classifier = FeedForwardNN(activation_function='tanh', input_size=10, layer_neurons=[5], output_size=2)
    x = np.random.uniform(low=0, high=1,size=(2,10,))#* 255
    output, threshold_output = classifier.forward(x)
    loss_prime =  np.array([[0.58652073, 0.40298107, 1., 0.38419388,0.25782405, 0.54964009,0.,0.73728738, 0.73501868, 0.54607665]])
    loss_prime =  np.array([[0.58652073, 0.40298107]])
    classifier.backward(loss_prime)