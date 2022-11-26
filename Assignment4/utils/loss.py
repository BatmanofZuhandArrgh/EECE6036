import numpy as np

def bce(y_truth, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1- 1e-7)
    return y_truth* np.log(y_pred) + (1 - y_truth) * np.log(1-y_pred)

def bce_prime(y_truth, y_pred):
    raise NotImplementedError

def mse(y_truth, y_pred):
    return np.mean((y_truth-y_pred) ** 2)

def mse_prime(y_truth, y_pred):
    #I know in the slides is just -(y_truth - ypred) for the loss_prime element, 
    # but this is just a scaled loss compare to that, so no difference
    return np.mean(-2*(y_truth-y_pred)/y_truth.size, axis = 0) #len(y_truth is num_sample), reduced by num_sample/batch
    
def get_loss(loss_type):
    if loss_type == 'mse':
        return mse, mse_prime
    elif loss_type == 'bce':
        return bce, bce_prime
    else:
        raise ValueError

if __name__ == '__main__':
    y_truth = np.array([[0,0,1,0,0],[0,0,0,0,0]])
    y_pred  = np.array([[0.58652073, 0.40298107, 1., 0.38419388,0.25782405], [0.54964009,0.,0.73728738, 0.73501868, 0.54607665]])
    # loss = mse(y_truth, y_pred)
    # print(loss, loss.shape)
    loss_prime = mse_prime(y_truth, y_pred)
    print(loss_prime, loss_prime.shape)