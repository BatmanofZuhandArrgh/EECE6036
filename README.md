# EECE6036
Intelligent System

Assigment 1:
Assignment1/spike.py: Re-Implementation of neuron firing in matlab fire and visualization code
Assignment1/neuron.py: Implementataion of a neuron class.
- See below for specific functions

1. To install
'''
pip install -r Assignment1/requirements.txt
'''

2. To generate regular spiking plots.
'''
python Assignment1/plot_regular_spiking.py
'''

3. To generate fast spiking plots.
'''
python Assignment1/plot_fast_spiking.py
'''

4. To generate plots of membrane potentials of a 2-neuron network
'''
python Assigment1/plot_2_neural_network.py
'''




Assigment 2:
1. To install
'''
cd EECE6036
pip install -r Assignment1/requirements.txt
'''
2. Run python Assignment2/parse_data_file.py to process and output data files train_img.txt, train_lab.txt (test and chal as well)
3. To run the problem 1 simulation, run 
'''
pip install -r Assignment2/Problem1.py
'''
4. To run the problem 2 simulation, run 
'''
pip install -r Assignment2/Problem2.py
'''

Assigment 3:
1. To install
'''
cd EECE6036
pip install -r Assignment1/requirements.txt
'''

2. Run python Assignment3/parse_data_file.py to process and output data files dataset/train_4000img.txt and dataset/train_4000lab.txt (test and chal as well)

3. To run the problem 1, edit the hyper-params in Assignment3/train.py 
'''
cd Assignment3
python -m train_prob1
'''

4. To run the problem 2.1 and 2.2, edit the hyper-params in 
'''
cd Assignment3
#To train prob1 classifier
python -m train_prob1
#To train prob2 autoencoder
python -m train_prob2
'''