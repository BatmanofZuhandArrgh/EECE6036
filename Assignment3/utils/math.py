import numpy as np
def get_onehot(num_array, num_class):
    onehot =  np.array([0 for x in range(num_class * num_array.shape[0])]).reshape((num_array.shape[0], num_class))
    for i, num in enumerate(num_array):
        onehot[i][num] = 1
    return onehot

if __name__ == '__main__':
    a = get_onehot(num_array=np.array([0,1,2,3,4]), num_class=10)
    print(a, a.shape)