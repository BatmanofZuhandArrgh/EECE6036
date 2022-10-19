from eval import get_stats
from perceptron_network import PerceptronNetwork
from trainer import Trainer

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

if __name__ == '__main__':
    #Create trainer
    trainer = Trainer(PerceptronNetwork, lr= 0.005)
    trainer.save_weights('./Assignment2/weights/Prob2_pre')

    #Test on test set pre-training
    pre_test_pred = trainer.test()
    test_truth = trainer.test_lab
    _, _, _, _,  precision_pre, recall_pre, f1_pre = get_stats(pre_test_pred, test_truth)

    #Train network for several epochs
    train_error_fractions, test_error_fractions = trainer.train_val(num_epochs=200)   
    plt.plot(train_error_fractions)
    plt.plot(test_error_fractions)
    plt.title("Perceptron Network's errors while training")
    plt.xlabel('epochs')
    plt.ylabel('error fraction')
    plt.legend(['train_error_fractions', 'test_error_fractions'])
    plt.savefig('./Assignment2/output/prob2/training_error.png')

    #Save weights
    trainer.save_weights('./Assignment2/weights/Prob2_post')

    #Test on test set post-training
    post_test_pred = trainer.test()
    _, _, _, _,  precision_post, recall_post, f1_post = get_stats(post_test_pred, test_truth)
    stats_df = pd.DataFrame({
            'pre-training' : [precision_pre, recall_pre, f1_pre],
            'post-training' : [precision_post, recall_post, f1_post]
            },index = ['precision', 'recall', 'f1 score']
        )

    print(stats_df)
    fig = plt.figure(figsize=(15,10))
    stats_df.plot(kind='bar')
    plt.title("Precision, recall and f1 of Perceptron on test set before and after training")
    plt.xticks(rotation = 0)
    plt.savefig('./Assignment2/output/prob2/pre_rec_f1.png')

    #Concat 2 output weights heatmap
    preweight = cv2.resize(cv2.imread('./Assignment2/weights/Prob2_pre.png'), (256,256))
    postweight = cv2.resize(cv2.imread('./Assignment2/weights/Prob2_post.png'), (256,256))
    side_by_side = np.concatenate([preweight, postweight], axis=1)
    cv2.imwrite('./Assignment2/output/prob2/weights_compare.png', side_by_side)

    #Get challenge output:
    trainer.challenge(csv_path='./Assignment2/output/prob2/challenge.csv')