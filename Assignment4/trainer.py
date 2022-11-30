import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.eval import get_error_fraction, get_confusion_matrix
from parse_data_file import read_img_from_txt, read_lab_from_txt
from utils.math import get_onehot
from network.feedforward_network import FeedForwardNN

class Trainer:
    def __init__(
        self,
        train_img_path = '../dataset/train_4000imgs.txt', 
        train_lab_path = '../dataset/train_4000labs.txt',
        test_img_path  = '../dataset/test_1000imgs.txt',
        test_lab_path  = '../dataset/test_1000labs.txt',
        experiment_name = "exp_sample",
        prob = 'prob1'
    ):
      
        self.network = None
        self.train_img_path = train_img_path  
        self.train_lab_path = train_lab_path  
        self.test_img_path = test_img_path    
        self.test_lab_path = test_lab_path    
        self.exp = experiment_name
        self.prob = prob
        self.output_dir = f'./output/{prob}/{self.exp}'
        os.makedirs(self.output_dir, exist_ok=True)

        self.train_img = None
        self.train_lab = None
        self.test_img = None
        self.test_lab = None

        #Read data set
        self.train_img = read_img_from_txt(self.train_img_path)
        self.train_lab = np.array(read_lab_from_txt(self.train_lab_path))
        self.test_img  = read_img_from_txt(self.test_img_path)
        self.test_lab  = np.array(read_lab_from_txt(self.test_lab_path))

        self.loss = []
        self.train_error_fraction = []
        self.test_error_fraction = []

    def init_network(self, network):
        self.network = network

    def shuffle_trainset(self):
        indices = np.arange(self.train_img.shape[0])
        np.random.shuffle(indices)
        self.train_img = self.train_img[indices]

        self.train_lab = self.train_lab[indices]

    def train(self, num_epochs = 200, batch_size =1, early_stopping = 5, min_error_diff = 0.003):
        test_errors = []
        for epoch in tqdm(range(num_epochs)):
            self.shuffle_trainset()
            loss = 0
            if epoch% 2 == 0 or epoch == num_epochs - 1:
                train_output, train_error_fraction= self.test(dataset='train')
                test_output, test_error_fraction = self.test(dataset='test')
                self.train_error_fraction.append(train_error_fraction)
                self.test_error_fraction.append(test_error_fraction)
                self.network.clear_input()
                
                #Early stopping
                test_errors.append(test_error_fraction)
                test_error_diffs = list(np.array(test_errors[:-1]) - np.array(test_errors[1:]))
       
                # if epoch == num_epochs -1 or save_epoch:
                os.makedirs(f'./{self.output_dir}/{epoch}', exist_ok=True)
                get_confusion_matrix(
                    y_truth=self.train_lab,
                    y_pred=train_output,
                    save=True,
                    save_path=f'./{self.output_dir}/{epoch}/train_confusion_matrix.csv',
                    num_cls=self.network.output_size
                )

                get_confusion_matrix(
                    y_truth=self.test_lab,
                    y_pred=test_output,
                    save=True,
                    save_path=f'./{self.output_dir}/{epoch}/test_confusion_matrix.csv',
                    num_cls=self.network.output_size
                )    
                self.save_weights()

                if np.all(np.array(test_error_diffs[-early_stopping:]) - min_error_diff < 0) and len(test_error_diffs) !=0:
                    print('\n')
                    print('Train error: ',train_error_fraction,' Test error: ',test_error_fraction)
                    break

            for img_idx in range(int(self.train_img.shape[0]//batch_size) -1):
                cur_input = self.train_img[img_idx*batch_size:(img_idx+1)*batch_size, :]

                cur_input_lab = get_onehot(self.train_lab[img_idx*batch_size:(img_idx+1)*batch_size], num_class=self.network.output_size)
                cur_output, cur_threshold_output = self.network.forward(cur_input)
                loss += self.network.loss(cur_input_lab, cur_threshold_output)
                loss_prime = self.network.loss_prime(cur_input_lab, cur_threshold_output)
                self.network.backward(loss_prime)

            self.loss.append(loss/self.train_img.shape[0])
        
        self.save_loss_plot()
        self.save_errorfraction_plot()
            
    def train_reconstructor(self, num_epochs = 200, batch_size =1, early_stopping = 2, min_loss_diff = 0.00001):
        train_losses= []
        test_losses = []
        for epoch in tqdm(range(num_epochs)):
            self.shuffle_trainset()

            loss = 0
            for img_idx in range(int(self.train_img.shape[0]//batch_size) -1):
                cur_input = self.train_img[img_idx*batch_size:(img_idx+1)*batch_size, :]

                cur_output, _ = self.network.forward(cur_input)
                loss += self.network.loss(cur_input, cur_output)
                loss_prime = self.network.loss_prime(cur_input, cur_output)
                self.network.backward(loss_prime)

            self.loss.append(loss/int(self.train_img.shape[0]//batch_size))
        
            if epoch% 10 == 0 or epoch == num_epochs - 1:
                train_loss = self.test_reconstructor(dataset='train')
                test_loss = self.test_reconstructor(dataset='test')
                
                #Early stopping
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                test_loss_diffs = list(np.array(test_losses[:-1]) - np.array(test_losses[1:]))
                print('Train loss: ',train_loss,' Test error: ',test_loss)

                self.save_weights()

                if np.all(np.array(test_loss_diffs[-early_stopping:]) - min_loss_diff < 0) and len(test_loss_diffs) !=0:
                    print(f'\n Early stopping at epoch {epoch}:')
                    print('Train loss: ',train_loss,' Test error: ',test_loss)
                    self.save_loss_plot(test_losses)
                    break

        print('\n')        
        print('Train loss: ',train_loss,' Test loss: ',test_loss)
        self.save_loss_plot(train_losses, test_losses)

    def test_reconstructor(self, img_path = None, lab_path = None, dataset = 'test', batch_size = 100):
        if dataset == 'test':
            test_img = self.test_img
        elif dataset =='train':
            test_img = self.train_img
        elif dataset == 'new':
            test_img = read_img_from_txt(img_path) 
        else:
            raise ValueError     
        loss = 0
        for img_idx in range(int(test_img.shape[0]//batch_size)):
            cur_input = test_img[img_idx*batch_size:(img_idx+1)*batch_size, :]
            cur_output, _ = self.network.forward(cur_input)
            loss += self.network.loss(cur_input, cur_output)

        loss = loss/int(test_img.shape[0]//batch_size)
        self.network.clear_input()
        return loss

    def test(self, img_path = None, lab_path = None, dataset = 'test', batch_size = 1):
        if dataset == 'test':
            test_img = self.test_img
            test_lab = self.test_lab
        elif dataset =='train':
            test_img = self.train_img
            test_lab = self.train_lab  
        elif dataset == 'new':
            test_img = read_img_from_txt(img_path)
            if lab_path is not None: 
                test_lab = read_lab_from_txt(lab_path)
            else: 
                test_lab = np.array([int(os.path.basename(img_path).split('_')[0]) for x in range(100)])
        else:
            raise ValueError         

        test_output = []
        for img_idx in range(int(test_img.shape[0]//batch_size)):
            cur_input = test_img[img_idx*batch_size:(img_idx+1)*batch_size, :]
            cur_output, cur_threshold_output = self.network.forward(cur_input)
            test_output.extend(list(np.argmax(cur_output, axis=1)))
        
        error_fraction = get_error_fraction(list(test_lab), test_output)
        self.network.clear_input()
        return test_output, error_fraction

    def infer(self, input):
        output, threshold_output = self.network.forward(input)
        self.network.clear_input()
        return output, threshold_output

    def save_loss_plot(self, train_loss = None, test_loss = None):
        fig = plt.figure(figsize=(10,5))

        loss = self.loss if train_loss is None else train_loss
        line, = plt.plot(range(len(loss)), loss, color = 'r', label = 'Train Loss',linewidth=0.5)
        plt.title('Training loss')

        if test_loss != None:
            line1, = plt.plot(range(len(test_loss)), test_loss, color = 'b', label = 'Test Loss',linewidth=0.5)
            plt.xlabel(xlabel='Epoch')
            plt.ylabel(ylabel='Loss')
            plt.legend(handles = [line, line1],labels =  ['Train loss', 'Test loss'], loc = 'upper left')

        fig.savefig(f'./output/{self.prob}/{self.exp}/loss.png', bbox_inches='tight', pad_inches=0)

    def save_errorfraction_plot(self):
        fig = plt.figure(figsize=(20,10))
        line, = plt.plot(range(len(self.train_error_fraction)), self.train_error_fraction, color = 'r', label = 'Train error',linewidth=0.5)
        line1, = plt.plot(range(len(self.test_error_fraction)), self.test_error_fraction, color = 'b', label = 'Test error',linewidth=0.5)
        plt.xlabel(xlabel='Epoch')
        plt.ylabel(ylabel='Error fraction')
        plt.legend(handles = [line, line1],labels =  ['Train error', 'Test error'], loc = 'upper left')
        plt.title('Train and test set error fraction across epochs')
        fig.savefig(f'./output/{self.prob}/{self.exp}/TrainingError.png', bbox_inches='tight', pad_inches=0)
    
    def save_weights(self):        
        save_path = f'{self.output_dir}/weights'
        os.makedirs(save_path, exist_ok=True)
        self.network.save_weight(save_path)

    def load_weights(self, path = None):
        load_path = f'{self.output_dir}' if path is None else path
        self.network.load_weight(load_path)

if __name__ == "__main__":
    trainer = Trainer()
    classifier = FeedForwardNN(
        input_size=784, output_size=10,
        layer_neurons=[200],
        activation_function='tanh', 
        output_activation='sigmoid',
        loss_function = 'mse',
        )
    trainer.init_network(classifier)
    trainer.train(num_epochs=40, batch_size=1)   
    # _, error = trainer.test('test')
    # # print(error)

    # trainer.save_weights()
    # trainer.load_weights()
    # _, error = trainer.test('test')
    # # print(error)
