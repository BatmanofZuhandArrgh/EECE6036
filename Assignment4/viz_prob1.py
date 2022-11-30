from pprint import pprint
from trainer import Trainer
from network.feedforward_network import FeedForwardNN
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    hyper_params = {
        'case0':{
            'hidden_neuron': 200,
            'lr': 0.01,
            'momentum': 0.5,
            'L': 0.25,
            'H': 0.75,
            'num_epochs': 400,
            'freeze_first_layer': False,
            'load_first_weight': False,
            'weight_path': './output/prob1/case0/weights'
        },
        'case1':{
            'hidden_neuron': 200,
            'lr': 0.01,
            'momentum': 0.5,
            'L': 0.25,
            'H': 0.75,
            'num_epochs': 120,
            'freeze_first_layer': False,
            'load_first_weight': False,
            'weight_path': './output/prob1/case1/weights'
        }
    }

    exp = 'case0'
    trainer0 = Trainer(experiment_name=exp)
    classifier0 = FeedForwardNN(
        input_size=784, output_size=10,
        layer_neurons=[hyper_params[exp]['hidden_neuron']],
        activation_function='tanh', 
        output_activation='sigmoid',
        loss_function = 'mse',
        lr=hyper_params[exp]['lr'],
        H=hyper_params[exp]['H'],
        L=hyper_params[exp]['L'],
        momentum=hyper_params[exp]['momentum'],
        load_first_weight=hyper_params[exp]['load_first_weight'],
        freeze_first_layer=hyper_params[exp]['freeze_first_layer'],
        weight_path=hyper_params[exp]['weight_path']
        )
    trainer0.init_network(classifier0)

    exp = 'case1'
    trainer1 = Trainer(experiment_name=exp)
    classifier1 = FeedForwardNN(
        input_size=784, output_size=10,
        layer_neurons=[hyper_params[exp]['hidden_neuron']],
        activation_function='tanh', 
        output_activation='sigmoid',
        loss_function = 'mse',
        lr=hyper_params[exp]['lr'],
        H=hyper_params[exp]['H'],
        L=hyper_params[exp]['L'],
        momentum=hyper_params[exp]['momentum'],
        load_first_weight=hyper_params[exp]['load_first_weight'],
        freeze_first_layer=hyper_params[exp]['freeze_first_layer'],
        weight_path=hyper_params[exp]['weight_path']
        )
    trainer1.init_network(classifier1)


    #Visualize loss across class
    loss_dict = {
        'full_dataset': {}
    }
    _, loss_dict['full_dataset']['test_0'] = trainer0.test(dataset='test')
    _, loss_dict['full_dataset']['test_1'] = trainer1.test(dataset='test')
    _, loss_dict['full_dataset']['train_0'] = trainer0.test(dataset='train')
    _, loss_dict['full_dataset']['train_1'] = trainer1.test(dataset='train')
    for i in range(0, 10):
        loss_dict[str(i)] = {}
        _, loss_dict[str(i)]['test_0'] = trainer0.test(dataset='new', img_path=f'../dataset/{str(i)}_test_100imgs.txt')
        _, loss_dict[str(i)]['test_1'] = trainer1.test(dataset='new', img_path=f'../dataset/{str(i)}_test_100imgs.txt')
        _, loss_dict[str(i)]['train_0'] = trainer0.test(dataset='new', img_path=f'../dataset/{str(i)}_train_400imgs.txt')
        _, loss_dict[str(i)]['train_1'] = trainer1.test(dataset='new', img_path=f'../dataset/{str(i)}_train_400imgs.txt')
    pprint(loss_dict)

    keys = sorted([key for key in loss_dict.keys()])
    stats_df = pd.DataFrame({
            'Case I: train_set' : [loss_dict[key]['train_0'] for key in keys],
            'Case I: test_set' : [loss_dict[key]['test_0'] for key in keys],
            'Case II: train_set': [loss_dict[key]['train_1'] for key in keys],
            'Case II: test_set': [loss_dict[key]['test_1'] for key in keys],
            },index = keys
        )
    print(stats_df)
    
    fig = plt.figure(figsize=(15,10))
    stats_df.plot(kind='bar')
    plt.title("Reconstruction loss for full dataset and each digit by train set and test set")
    plt.xticks(rotation = 25)
    plt.savefig(f'./output/prob1/error_comparison.png')