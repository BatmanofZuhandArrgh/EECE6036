from pprint import pprint
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

from trainer import Trainer
from parse_data_file import read_img_from_txt
from network.autoencoder_network import AutoEncoder

def viz_neurons(array, num_neurons = 20, path = None, network_name = 'autoencoder'):
    fig, axs = plt.subplots(4, 5, figsize=(10, 10))
    fig.suptitle(f'Neurons of fist layer of the {network_name}')
    row = 0
    col = 0
    for i in range(num_neurons):
        index = random.randint(0,array.shape[1]-1)
        cur_sample = array[:, index]
        input_img = (cur_sample*255).reshape(28,28) 
        axs[row][col].imshow(input_img.T)
        col += 1
        if col > 4:
            col = 0
            row +=1

    fig.savefig(path, bbox_inches='tight', pad_inches=0.05)

if __name__ == "__main__":

    hyper_params = {
        'exp0':{
            'hidden_neuron': 200,
            'lr': 0.01,
            'momentum': 0.5,
            'num_epochs': 210
        },
        'exp1':{
            'hidden_neuron': 200,
            'lr': 0.01,
            'momentum': 0.5,
            'num_epochs': 400
        },
        'exp2':{
            'hidden_neuron': 200,
            'lr': 0.0025,
            'momentum': 0.5,
            'num_epochs': 810
        },
    }
    pprint(hyper_params)
    exp = 'exp2'
    output_path = f'./output/prob2/{exp}'
    trainer = Trainer(
        experiment_name=exp,
        prob='prob2'
        )
    reconstructor = AutoEncoder(
        input_size=784,
        layer_neurons=[hyper_params[exp]['hidden_neuron']],
        activation_function='tanh', 
        output_activation='sigmoid',
        loss_function = 'mse',
        lr=hyper_params[exp]['lr'],
        momentum=hyper_params[exp]['momentum'],
        )

    trainer.init_network(reconstructor)
    # trainer.load_weights() #Continue training
    trainer.train_reconstructor(
        num_epochs=hyper_params[exp]['num_epochs'], 
        batch_size=1)   
    
    #Visualize input and output of different samples
    train_img_path = '../dataset/test_1000imgs.txt'
    testing_samples = read_img_from_txt('../dataset/test_1000imgs.txt')
    
    fig, axs = plt.subplots(2, 8, figsize=(10, 3.5))
    fig.suptitle('Input and output of the autoencoder')

    for i in range(8):
        index = random.randint(0,testing_samples.shape[0]-1)
        cur_sample = testing_samples[index, :]
        output, _ = trainer.infer(cur_sample)
        
        input_img = (cur_sample*255).reshape(28,28) 
        axs[0][i].imshow(input_img.T)
        # plt.show()
        
        output_img = (output *255).reshape(28,28)
        axs[1][i].imshow(output_img.T)
        # plt.show()
    axs[0][4].set_title('Input from MNIST Test set')
    axs[1][4].set_title('Output through AutoEncoder')

    fig.savefig(f'{output_path}/sample_comparison.png', bbox_inches='tight', pad_inches=0.05)
    
    #Visualize neurons
    prob2_weight_savepath = f'{output_path}/prob2_weights.png'
    weight_array = np.load(f'{output_path}/weights/weight_0.npy')
    viz_neurons(weight_array,path = prob2_weight_savepath)

    prob1_weight_savepath = f'./output/prob1/exp0/prob1_weights.png'
    weight_array = np.load(f'./output/prob1/exp0/weights/weight_0.npy')
    viz_neurons(weight_array,path = prob1_weight_savepath, network_name='classifier')

    #Visualize loss across class
    loss_dict = {
        'full_dataset': {}
    }
    loss_dict['full_dataset']['train'] = trainer.test_reconstructor(dataset='train')
    loss_dict['full_dataset']['test'] = trainer.test_reconstructor(dataset='test')
    for i in range(0, 10):
        loss_dict[str(i)] = {}
        loss_dict[str(i)]['train'] = trainer.test_reconstructor(dataset='new', img_path=f'../dataset/{str(i)}_train_400imgs.txt')
        loss_dict[str(i)]['test'] = trainer.test_reconstructor(dataset='new', img_path=f'../dataset/{str(i)}_test_100imgs.txt')
    pprint(loss_dict)
    keys = sorted([key for key in loss_dict.keys()])
    stats_df = pd.DataFrame({
            'train' : [loss_dict[key]['train'] for key in keys],
            'test' : [loss_dict[key]['test'] for key in keys]
            },index = keys
        )
    print(stats_df)
    
    fig = plt.figure(figsize=(15,10))
    stats_df.plot(kind='bar')
    plt.title("Reconstruction loss for full dataset and each digit by train set and test set")
    plt.xticks(rotation = 25)
    plt.savefig(f'./{output_path}/loss_comparison.png')