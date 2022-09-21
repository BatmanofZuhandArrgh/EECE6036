import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from neuron import Neuron
IA = 5
IB = 2

def input_current(constant_I_supply, weight, y_t):
    return constant_I_supply + weight * y_t

def plot_2Neurons(
    tspan, array,array1, 
    xlabel, ylabel, title, 
    miny, maxy, save_path
    ):

    fig = plt.figure(figsize=(10,5))
    line, = plt.plot(tspan, array, color = 'r', label = 'A',linewidth=0.5)
    line1, = plt.plot(tspan, array1, color = 'b', label = 'B',linewidth=0.5)

    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    
    if miny is not None and maxy is not None:
        plt.ylim([miny, maxy])
        plt.xlim([tspan[0], tspan[-1]])

    plt.legend(handles = [line, line1],labels =  ['A', 'B'], loc = 'upper left')

    plt.title(title)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

def main():
    steps = 1000
    tau = 0.25
    tspan = np.linspace(0,steps,int(steps/tau) + 1) #tspan is the simulation interval
    
    for W in tqdm([0,10,20,30,40]):
        neuronA = Neuron('CH', starting_time =0)
        neuronB = Neuron('CH', starting_time =0)

        for t in tspan:
            if t==0:
                yA, yB = 0,0
            else:
                yA = 1 if vA >= 30 else 0
                yB = 1 if vB >= 30 else 0

            Ia = IA - W * yB
            Ib = IB + W * yA
            
            neuronA.spike(cur_t=t, I_var = Ia)
            neuronB.spike(cur_t=t, I_var = Ib)

            vA = neuronA.get_v()
            vB = neuronB.get_v()

        VVA = neuronA.get_vv()
        VVB = neuronB.get_vv()
        
        plot_2Neurons(
            tspan,
            array  = VVA,
            array1 = VVB,
            xlabel = "time step", 
            ylabel = 'V_m', 
            title = f'2_neuron_network at W = {W}', 
            miny = -90, 
            maxy = 50, 
            save_path = f'Assignment1/2_neuron_network/potential_2_neuron_network_W{W}.png'
        )
        

if __name__ == '__main__':
    main()