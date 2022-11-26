import numpy as np
import pandas as pd

def get_error_fraction(y_truth, y_pred):
    incorrect = 0
    for a,b in zip(y_truth, y_pred):
        if a!= b:
            incorrect += 1
    return incorrect/len(y_truth)

def get_confusion_matrix(y_truth, y_pred, save = True, save_path = None, num_cls = 10):
    confusion_mat = np.zeros(shape=(num_cls, num_cls))
    for i in range(len(y_truth)):
        confusion_mat[y_truth[i]][y_pred[i]] += 1
    
    if save:
        df = pd.DataFrame(confusion_mat)
        df.to_csv(save_path)
    #ylabel: truth
    #xlabel: pred
    return confusion_mat

if __name__ == '__main__':
    y_truth = [1,2,3,4]
    y_pred  = [3,2,3,4]
    # print(get_error_fraction(np.array([1,2,3,4]), np.array([3,2,3,4])))
    get_confusion_matrix(y_truth, y_pred, num_cls=5)