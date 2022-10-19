import numpy as np

class NeuralNetwork:
    def __init__(self, weight_path = None, num_neuron = 784, lr = 0.01) -> None:
        self.num_neuron = num_neuron

        if weight_path is None:
            self.weights = np.random.uniform(low=0, high=0.5, size=(num_neuron,))
        else:
            self.weights = np.load(weight_path)

        self.lr = lr

    def forward(self, input):
        raise NotImplementedError

    def update(self, input, output, ground_truth = None):
        raise NotImplementedError