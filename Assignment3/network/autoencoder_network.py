import numpy as np
from utils.loss import mse_prime
from network.neural_network import NeuralNetwork

class AutoEncoder(NeuralNetwork):
    def __init__(self, weight_path=None, layer_neurons=[200], lr=0.01, input_size=784, activation_function=None, output_activation='sigmoid', H=0.75, L=0.25, momentum=0.5, loss_function='mse'):
        super().__init__(
            weight_path, 
            layer_neurons, 
            lr, 
            input_size, 
            output_size = input_size, 
            activation_function = activation_function,
            output_activation = output_activation,
            momentum =momentum, 
            loss_function = loss_function)

        self.input = []
        self.layer_types = []

        for i in range(len(self.weights)):
            self.layer_types.append('fc')
            if i == (len(self.weights) - 1):
                self.layer_types.append('output_act')            
            else:
                self.layer_types.append('hidden_act')

        # for weight in self.weights:
        #     print(weight.shape)

    def forward(self, x):
        for i in range(len(self.weights)):
            self.input.append(x)  
            # print(x.shape, self.weights[i].shape, self.biases[i].shape)
            x = np.dot(x, self.weights[i]) + self.biases[i]
            self.input.append(x)

            if i == (len(self.weights) - 1):
                x = self.out_act(x)
                # print(x, x.shape)
            else:
                x = self.f(x)
        
        return x, None

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
    autoencoder = AutoEncoder(activation_function='tanh', input_size=784, layer_neurons=[200])
    x = np.random.uniform(low=0, high=1,size=(2,784,))#* 255
    output, _ = autoencoder.forward(x)
    print(x.shape, output.shape)
    
    loss_prime = mse_prime(x,  output) 
    autoencoder.backward(loss_prime)