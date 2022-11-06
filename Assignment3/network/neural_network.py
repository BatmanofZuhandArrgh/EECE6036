import numpy as np

class NeuralNetwork:
    def __init__(self, weight_path = None, layer_neurons = [200], lr = 0.01, input_size = 784, num_class = 10) -> None:
        '''
        Input:
        - Weight path: path to npy file that contains weights
        - Layer_neurons: list of ints, number of neurons that each hidden layer has, excluding input and output layer
        - lr: learning rate
        - input_size
        - num_cls: int, output size
        '''
        self.layer_neurons =  [input_size] + layer_neurons + [num_class]

        self.weights = []
        for index in range(len(self.layer_neurons)-1):
            layer_input_size = self.layer_neurons[index]
            layer_output_size= self.layer_neurons[index+1]

            if weight_path is None:
                cur_weight = np.random.uniform(low=0, high=0.5, size=(layer_input_size, layer_output_size))
            else:
                print('load weight path')
                raise NotImplementedError
                self.weights = np.load(weight_path)

            self.weights.append(cur_weight)
        # for weight in self.weights:
        #     print(weight.shape)
        
        self.lr = lr

    def forward(self, input):
        raise NotImplementedError

    def update(self, input, output, ground_truth = None):
        raise NotImplementedError

if __name__ == "__main__":
    sample_network = NeuralNetwork(layer_neurons=[200, 242, 12])
