import numpy as np
import os
from utils.activation import get_act
from utils.loss import get_loss

class NeuralNetwork:
    def __init__(self, weight_path = None, layer_neurons = [200], lr = 0.01, input_size = 784, output_size = 10,
     activation_function = None,
     output_activation = None,
     momentum = 0.5,
     loss_function = None,
     ) -> None:
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
        self.input_size = input_size
        self.output_size = output_size
        self.layer_neurons =  [input_size] + layer_neurons + [output_size]

        for index in range(len(self.layer_neurons)-1):
            layer_input_size = self.layer_neurons[index]
            layer_output_size= self.layer_neurons[index+1]

            if weight_path is None:
                # np.random.seed(42) 
                cur_weight = np.random.rand(layer_input_size, layer_output_size) - 0.5
                # np.random.seed(0) 
                cur_bias   = np.random.rand(1, layer_output_size) - 0.5
            else:
                print('load weight path')
                self.load_weight(weight_path)

            self.weights.append(cur_weight)
            self.biases.append(cur_bias)

            self.act_func = activation_function
            self.out_act_func = output_activation
            self.loss_func = loss_function

            self.f, self.f_prime = get_act(self.act_func)
            self.out_act, self.out_act_prime = get_act(output_activation)
            self.loss, self.loss_prime = get_loss(self.loss_func)

        # for weight in self.weights:
        #     print(weight.shape)
        
        self.lr = lr
        self.momentum = momentum
        self.weight_gradients = None #Saving for momentum

    def forward(self, input):
        raise NotImplementedError

    def backward(self, loss):
        raise NotImplementedError
    
    def layer_backprop(self, layer_input, layer_output_error, weight_index, layer_type):
        self.weight_gradients = [0 for x in self.layer_types if x == 'fc']

        if layer_type == 'fc':
            # print(layer_output_error.shape, self.weights[weight_index].T.shape)
            layer_input_error = np.dot(layer_output_error, self.weights[weight_index].T)

            if len(layer_input.shape) == 1:
                # Adjust first input
                layer_input = np.expand_dims(layer_input, axis=0)
            # print(layer_input.T.shape, layer_output_error.shape)

            weight_error= np.dot(layer_input.T,layer_output_error)
            # print('layer',layer_output_error, layer_output_error.shape)
            bias_error  = np.mean(layer_output_error, axis = 0, keepdims=True) 
            # print('bias', bias_error, bias_error.shape)

            self.weight_gradients[weight_index] = self.lr * weight_error + self.momentum * self.weight_gradients[weight_index] 
    
            self.weights[weight_index] -= self.weight_gradients[weight_index]
            # print(self.biases[weight_index].shape, bias_error.shape)
            self.biases[weight_index]  -= self.lr * bias_error

        elif layer_type == 'hidden_act':
            layer_input_error = self.f_prime(layer_input)* layer_output_error
        elif layer_type == 'output_act':
            layer_input_error = self.out_act_prime(layer_input)* layer_output_error

        return layer_input_error
    
    def save_weight(self, save_path):
        for i in range(len(self.weights)):
            np.save( f'{save_path}/weight_{i}.npy', self.weights[i])
        for i in range(len(self.biases)):
            np.save( f'{save_path}/bias_{i}.npy', self.biases[i])

    def load_weight(self, load_path):
        self.biases = []
        self.weights = []
        npy_files = [x for x in os.listdir(load_path) if '.npy' in x]
        weight_files = sorted([os.path.join(load_path,x) for x in npy_files if 'weight_' in x])
        bias_files = sorted([os.path.join(load_path,x) for x in npy_files if 'bias_' in x])
        for i in range(len(weight_files)):
            self.biases.append((np.load(bias_files[i])))
            self.weights.append((np.load(weight_files[i])))

if __name__ == "__main__":
    sample_network = NeuralNetwork(layer_neurons=[200])
