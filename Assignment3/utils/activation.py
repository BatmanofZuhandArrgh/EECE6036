import numpy as np
np.set_printoptions(suppress=True)

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x) :
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return (np.exp(x) -  np.exp(-x))/ (np.exp(x) + np.exp(-x))

def tanh_prime(x):
    return 1 - tanh(x) **2 

def relu(x):
    return np.max(np.zeros_like(x), x)

def relu_prime(x):
    return 1 if x >= 0 else 0

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()

def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))

def get_act(act_type):
    if act_type == 'relu':
        f = relu
        f_prime = relu_prime
    elif act_type == 'sigmoid':
        f = sigmoid
        f_prime = sigmoid_prime
    elif act_type == 'tanh':
        f = tanh
        f_prime = tanh_prime
    else:
        raise ValueError
    return f, f_prime


