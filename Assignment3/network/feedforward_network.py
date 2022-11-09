import numpy as np
from network.neural_network import NeuralNetwork
from utils.math_utils import *
np.set_printoptions(suppress=True)


class FeedForwardNN(NeuralNetwork):
    def __init__(self, weight_path=None, layer_neurons=[200], lr=0.01, input_size=784, num_class=10, activation_function=None, 
        output_activation = 'sigmoid',
        H = 0.75, L =0.25
        ):
        super().__init__(weight_path, layer_neurons, lr, input_size, num_class, activation_function, output_activation)
        self.H, self.L = H,L
        self.input = []
        self.layer_types = []

    def thresholding(self, x):
        new_x = []
        for e in x[0]:
            if e >= self.H: new_x.append(1)
            elif e <= self.L: new_x.append(0)
            else: new_x.append(e)
        return np.array(new_x)

    def forward(self, x):
        for i in range(len(self.weights)):
            self.input.append(x)
            self.layer_types.append('fc')
            x = np.dot(x, self.weights[i]) + self.biases[i]
            # print(x, x.shape)
            self.input.append(x)
            if i == (len(self.weights) - 1):
                self.layer_types.append('output_act')
                x = self.out_act(x)
                # print(x, x.shape)
                x = self.thresholding(x) 
            else:
                self.layer_types.append('hidden_act')
                x = self.f(x)

            # print(x, x.shape)
        return x

    def backward(self, loss):
        layer_output_error = loss
        weight_index = -1
        for i in range(-1, -len(self.layer_types) -1, -1):
            print(i, self.layer_types[i])
            layer_output_error = self.layer_backprop(
                layer_input=self.input[i],
                layer_output_error=layer_output_error,
                weight_index=weight_index,
                layer_type=self.layer_types[i]
            )
            if self.layer_types[i] == 'fc':
                weight_index -= 1

        self.input = []
        self.layer_types = []
        
      
if __name__ == '__main__':
    classifier = FeedForwardNN(activation_function='tanh')
    x = np.random.uniform(low=0, high=1,size=(784,))#* 255
    classifier.forward(x)
    classifier.backward(np.array([0.58652073, 0.40298107, 1., 0.38419388,0.25782405, 0.54964009,0.,0.73728738, 0.73501868, 0.54607665]))