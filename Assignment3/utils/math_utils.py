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

def binary_cross_entropy(y_truth, y_pred):
    #For single values
    y_pred = np.clip(y_pred, 1e-7, 1- 1e-7)
    return y_truth* np.log(y_pred) + (1 - y_truth) * np.log(1-y_pred)

if __name__ == '__main__':
    y_truth = np.array([[0,0,1,0,0],[0,0,0,0,0]])
    y_pred  = np.array([[0.58652073, 0.40298107, 1., 0.38419388,0.25782405], [0.54964009,0.,0.73728738, 0.73501868, 0.54607665]])
    print(binary_cross_entropy(y_truth, y_pred))