from reinforced_network import ReinforcedNetwork, plot_eval, theta_conversion
from trainer import Trainer

import cv2
import numpy as np
if __name__ == '__main__':
    #Create trainer
    trainer = Trainer(ReinforcedNetwork, lr = 0.01)
    trainer.save_weights('./Assignment2/weights/Prob1_pre')

    #Train network for several epochs
    trainer.train(num_epochs=40)   
    
    #Save weights
    trainer.save_weights('./Assignment2/weights/Prob1_post')

    #Test on test set
    test_output = trainer.test()
    test_truth = trainer.test_lab
    
    #Calculate stats at each theta values and plot
    test_preds = theta_conversion(test_output, theta_range=range(0,41))
    optimal_theta = plot_eval(test_preds, ground_truth=test_truth)

    #Concat 2 output weights heatmap
    preweight = cv2.resize(cv2.imread('./Assignment2/weights/Prob1_pre.png'), (256,256))
    postweight = cv2.resize(cv2.imread('./Assignment2/weights/Prob1_post.png'), (256,256))
    side_by_side = np.concatenate([preweight, postweight], axis=1)
    cv2.imwrite('./Assignment2/output/prob1/weights_compare.png', side_by_side)

    #Get challenge output:
    trainer.challenge(optimal_threshold=optimal_theta)