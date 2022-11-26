from pprint import pprint
from trainer import Trainer
from network.feedforward_network import FeedForwardNN

if __name__ == "__main__":

    hyper_params = {
        'exp0':{
            'hidden_neuron': 200,
            'lr': 0.01,
            'momentum': 0.5,
            'L': 0.25,
            'H': 0.75,
            'num_epochs': 80
        },
        'exp1':{
            'hidden_neuron': 150,
            'lr': 0.005,
            'momentum': 0.5,
            'L': 0.1,
            'H': 0.9,
            'num_epochs': 50
        },
        'exp2':{
            'hidden_neuron': 200,
            'lr': 0.005,
            'momentum': 0.5,
            'L': 0.1,
            'H': 0.9,
            'num_epochs': 70
        },
        'exp3':{
            'hidden_neuron': 200,
            'lr': 0.01,
            'momentum': 0.5,
            'L': 0.25,
            'H': 0.75,
            'num_epochs': 80
        },
        'exp4':{
            'hidden_neuron': 200,
            'lr': 0.01,
            'momentum': 0.5,
            'L': 0.1,
            'H': 0.9,
            'num_epochs': 80
        },
        'exp6':{
            'hidden_neuron': 200,
            'lr': 0.01,
            'momentum': 0.3,
            'L': 0.25,
            'H': 0.75,
            'num_epochs': 3
        },
        'exp7':{
            'hidden_neuron': 100,
            'lr': 0.01,
            'momentum': 0.5,
            'L': 0.25,
            'H': 0.75,
            'num_epochs': 80
        },
    }
    pprint(hyper_params)
    exp = 'exp7'
    trainer = Trainer(experiment_name=exp)
    classifier = FeedForwardNN(
        input_size=784, output_size=10,
        layer_neurons=[hyper_params[exp]['hidden_neuron']],
        activation_function='tanh', 
        output_activation='sigmoid',
        loss_function = 'mse',
        lr=hyper_params[exp]['lr'],
        H=hyper_params[exp]['H'],
        L=hyper_params[exp]['L'],
        momentum=hyper_params[exp]['momentum'],
        )

    trainer.init_network(classifier)
    trainer.train(
        num_epochs=hyper_params[exp]['num_epochs'], 
        batch_size=1)   
