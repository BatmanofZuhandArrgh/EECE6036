import numpy as np
from eval import get_stats
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork

class ReinforcedNetwork(NeuralNetwork):
    def __init__(self, weight_path=None, num_neuron=784, lr=0.01) -> None:
        super().__init__(weight_path, num_neuron, lr)

    def forward(self, input):
        return sum(self.weights * input)

    def update(self, input, output, ground_truth = None):
        cur_weights = self.weights
        self.weights = cur_weights + self.lr * ground_truth * (input - cur_weights)

def theta_conversion(raw_output, theta_range = range(0, 41)):
    output_lists = []
    for theta in theta_range:
        cur_output = [1 if x > theta else 0 for x in raw_output]
        output_lists.append(cur_output)

    return output_lists

def plot_eval(output_lists, ground_truth):
    precisions = []
    recalls = []
    f1s = []
    tps = []
    tns = []
    fps = []
    fns = []
    tprs = []
    fprs = []

    for output in output_lists:  
        tp, tn, fp, fn, precision, recall, f1 = get_stats(pred=output, truth = ground_truth)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        tprs.append(recall) #True Positive Rate
        fpr = fp/(fp + tn) if (fp + tn) != 0 else 0        
        fprs.append(fpr)
    
    #Saving precision, recall and f1 graph
    fig = plt.figure(figsize = (20,10))
    plt.plot(precisions)
    plt.plot(recalls)
    plt.plot(f1s)
    plt.title('Precision, Recall and F1 using simple reinforment paradigm')
    plt.xlabel('theta')
    plt.ylabel('Value')
    plt.legend(['precision', 'recall', 'f1'])
    plt.savefig('./Assignment2/output/prob1/pre_rec_f1.png')

    #Plot ROC curves
    fig = plt.figure(figsize=(20,10))
    plt.plot(fprs, tprs)
    plt.title('ROC curves using simple reinforment paradigm')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('./Assignment2/output/prob1/ROC_curves.png')

    #Picking the optimal threshold
    optimal_values = [tpr+1-fpr for tpr, fpr in zip (tprs, fprs)]
    optimal_index = np.argmax(optimal_values) 
    print(f"The optimal threshold value is theta = {optimal_index}, where TPR = {tprs[optimal_index]}, FPR = {fprs[optimal_index]}, and F1 = {f1s[optimal_index]}")
    return optimal_index
    
if __name__ == '__main__':
    reinforced_network = ReinforcedNetwork()
    # print(reinforced_network.weights)
