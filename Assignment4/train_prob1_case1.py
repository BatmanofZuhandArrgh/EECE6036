from pprint import pprint
from trainer import Trainer
from network.feedforward_network import FeedForwardNN

if __name__ == "__main__":

    hyper_params = {
        'case1':{
            'hidden_neuron': 200,
            'lr': 0.01,
            'momentum': 0.5,
            'L': 0.25,
            'H': 0.75,
            'num_epochs': 120,
            'freeze_first_layer': False,
            'load_first_weight': True,
            'weight_path': './output/prob1/exp_A3/weights'
        },
    }
    pprint(hyper_params)
    exp = 'case1'
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
        load_first_weight=hyper_params[exp]['load_first_weight'],
        freeze_first_layer=hyper_params[exp]['freeze_first_layer'],
        weight_path=hyper_params[exp]['weight_path']
        )

    trainer.init_network(classifier)
    trainer.train(
        num_epochs=hyper_params[exp]['num_epochs'], 
        batch_size=1)   
