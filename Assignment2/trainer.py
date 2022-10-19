import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from parse_data_file import read_img_from_txt, read_lab_from_txt
from reinforced_network import ReinforcedNetwork

class Trainer:
    def __init__(
        self,
        network_type,
        train_img_path = './dataset/train_img.txt', 
        train_lab_path = './dataset/train_lab.txt',
        test_img_path  = './dataset/test_img.txt',
        test_lab_path  = './dataset/test_lab.txt',
        chal_img_path  = './dataset/chal_img.txt',
        chal_lab_path  = './dataset/chal_lab.txt',
        lr = 0.01
    ):
      
        self.network = network_type(lr= lr)
        self.train_img_path = train_img_path  
        self.train_lab_path = train_lab_path  
        self.test_img_path = test_img_path    
        self.test_lab_path = test_lab_path    
        self.chal_img_path = chal_img_path    
        self.chal_lab_path = chal_lab_path    

        self.train_img = None
        self.train_lab = None
        self.test_img = None
        self.test_lab = None
        self.chal_img = None
        self.chal_lab = None

        #Read data set
        self.train_img = read_img_from_txt(self.train_img_path)
        self.train_lab = read_lab_from_txt(self.train_lab_path)
        self.test_img  = read_img_from_txt(self.test_img_path)
        self.test_lab  = read_lab_from_txt(self.test_lab_path)
        self.chal_img  = None
        self.chal_lab  = None

    def train(self, num_epochs = 40):

        for epoch in tqdm(range(num_epochs)):
            for img_idx in range(self.train_img.shape[0]):
                cur_input = self.train_img[img_idx, :]
                cur_input_lab = self.train_lab[img_idx]
                cur_output = self.network.forward(cur_input)
                self.network.update(cur_input, cur_output, cur_input_lab)

    def test(self):
        test_output = []
        for img_idx in range(self.test_img.shape[0]):
            cur_input = self.test_img[img_idx, :]
            cur_test_output = self.network.forward(cur_input)
            test_output.append(cur_test_output)

        return test_output

    def train_val(self, num_epochs):
        train_errors = []
        test_errors  = []
        for epoch in tqdm(range(num_epochs)):
            #Train
            num_train_error = 0
            for img_idx in range(self.train_img.shape[0]):
                cur_input = self.train_img[img_idx, :]
                cur_input_lab = self.train_lab[img_idx]

                cur_output = self.network.forward(cur_input)
                if cur_output != cur_input_lab: num_train_error += 1
                self.network.update(cur_input, cur_output, cur_input_lab)
            train_errors.append(num_train_error/self.train_img.shape[0])
            
            #Val 
            num_test_error = 0
            for img_idx in range(self.test_img.shape[0]):
                cur_input = self.test_img[img_idx, :]
                cur_input_lab = self.test_lab[img_idx]
                cur_output = self.network.forward(cur_input)
                if cur_output != cur_input_lab: num_test_error += 1
            test_errors.append(num_test_error/self.test_img.shape[0])

        return train_errors, test_errors


    def save_weights(self, save_path = './Assignment2/weights/weights.npy'):        
        np.save(save_path + '.npy', self.network.weights)
        graph = np.array([int(float(x)*255) for x in self.network.weights])
        graph = np.reshape(graph, (28, 28))
        plt.imsave(save_path + '.png',graph)

    def challenge(self, csv_path = './Assignment2/output/prob1/challenge.csv', optimal_threshold = None):
        self.chal_img = read_img_from_txt(self.chal_img_path)
        self.chal_lab = read_lab_from_txt(self.chal_lab_path)
        
        report_dict = {}
        report_dict['label'] = [0, 1]
        for num in range(2, 10):
            report_dict[num] = [0,0]
        
        for img_idx, img in enumerate(self.chal_img):
            output = self.network.forward(img)
            if optimal_threshold is not None:
                output = 1 if output > optimal_threshold else 0
            
            if output:
                report_dict[self.chal_lab[img_idx]][1] += 1
            else:
                report_dict[self.chal_lab[img_idx]][0] += 1


        report = pd.DataFrame(data = report_dict)
        report.to_csv(csv_path, index = False)
            

if __name__ == "__main__":
    trainer = Trainer(ReinforcedNetwork)
    trainer.train(num_epochs=1)   
    print(trainer.test())
