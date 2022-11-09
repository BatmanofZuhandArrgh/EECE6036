import numpy as np
from utils.math_utils import get_act

class NeuralNetwork:
    def __init__(self, weight_path = None, layer_neurons = [200], lr = 0.01, input_size = 784, output_size = 10,
     activation_function = None,
     output_activation = None) -> None:
        '''
        Input:
        - Weight path: path to npy file that contains weights
        - Layer_neurons: list of ints, number of neurons that each hidden layer has, excluding input and output layer
        - lr: learning rate
        - input_size
        - output_size: int, output num_cls
        '''

        self.weights = []
        self.biases  = []
        self.layer_neurons =  [input_size] + layer_neurons + [output_size]

        for index in range(len(self.layer_neurons)-1):
            layer_input_size = self.layer_neurons[index]
            layer_output_size= self.layer_neurons[index+1]

            if weight_path is None:
                cur_weight = np.random.rand(layer_input_size, layer_output_size) - 0.5
                cur_bias   = np.random.rand(1, layer_output_size) - 0.5
            else:
                print('load weight path')
                raise NotImplementedError
                self.weights = np.load(weight_path)

            self.weights.append(cur_weight)
            self.biases.append(cur_bias)

            self.act_func = activation_function
            self.out_act_func = output_activation

            self.f, self.f_prime = get_act(self.act_func)
            self.out_act, self.out_act_prime = get_act(output_activation)

        for weight in self.weights:
            print(weight.shape)
        
        self.lr = lr

    def forward(self, input):
        raise NotImplementedError

    def backward(self, loss):
        raise NotImplementedError
    
    def layer_backprop(self, layer_input, layer_output_error, weight_index, layer_type):
        if layer_type == 'fc':
            # print(layer_output_error.shape, self.weights[weight_index].T.shape)
            layer_input_error = np.dot(layer_output_error, self.weights[weight_index].T)

            if len(layer_input.shape) == 1:
                #Adjust first input
                layer_input = np.expand_dims(layer_input, axis=0)
            # print(layer_input.T.shape, layer_output_error.shape)

            weight_error= np.dot(layer_input.T,layer_output_error)
            bias_error  = layer_output_error 
            
            self.weights[weight_index] -= self.lr * weight_error
            self.biases[weight_index]  -= self.lr * bias_error
            
        elif layer_type == 'hidden_act':
            layer_input_error = self.f_prime(layer_input) * layer_output_error
        elif layer_type == 'output_act':
            layer_input_error = self.out_act_prime(layer_input) * layer_output_error

        return layer_input_error

if __name__ == "__main__":
    sample_network = NeuralNetwork(layer_neurons=[200])
