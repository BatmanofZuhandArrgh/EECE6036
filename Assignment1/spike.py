import numpy as np
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

#Inspired by in the paper by Izhikevich E.M. (2004) 
def membrane_potential(I_var, save, save_path,a,b,c,d, title):
    steps = 1000                 #This simulation runs for 1000 steps
    V=-64 
    u=b*V
    VV=[]
    uu=[]
    tau = 0.25                  #tau is the discretization time-step
    tspan = np.linspace(0,steps,int(steps/tau) + 1) #tspan is the simulation interval

    T1=0 #changed from 50                       #T1 is the time at which the step input rises
    spike_ts = []

    for t in tspan:
        if t>T1:
            I = I_var
        else:
            I = 0
        V = V + tau*(0.04*V**2+5*V+140-u+I)
        u = u + tau*a*(b*V-u)

        if V > 30:              #If this is a spike
            VV.append(30)       #VV is the time-series of membrane potentials
            V = c
            u = u + d
            spike_ts.append(1)  #Record a spike
        else:
            VV.append(V)
            spike_ts.append(0)  #Record a spike

        uu.append(u)
    
    plot_spike(tspan, array=VV, 
        title=title, 
        miny=-90, maxy = 40,
        xlabel='time step',
        ylabel='V_m',
        save = save,
        save_path=save_path,
        label = title
    )
        
    spikes_last_800_steps = spike_ts[int(1000*4*0.2):]
    R = sum(spikes_last_800_steps)/800
    return R

def plot_spike(tspan, array, label, xlabel, ylabel, title, miny, maxy, save_path, color = 'b', save = False, compare_array=None):
    fig = plt.figure(figsize=(10,5))
    line, = plt.plot(tspan, array, color = color, label = label,linewidth=0.5)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    
    if miny is not None and maxy is not None:
        plt.ylim([miny, maxy])
        plt.xlim([tspan[0], tspan[-1]])

    if compare_array is not None:
        array1 = np.load(compare_array)
        line1, = plt.plot(tspan, array1, color = 'm', label = 'regular_spiking',linewidth=0.5)
        plt.legend(handles = [line, line1],labels =  ['FS', 'RS'], loc = 'upper left')

    plt.title(title)
    # plt.show()
    if save:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

def plot_membrane_potential(num_trials, folder_name,a,b,c,d, compare_array = None):
    R = []

    for i0 in tqdm(range(num_trials)):
        save = False
        save_path = None
        title = f'{folder_name}_I{i0}'

        if i0 in [1,10, 20, 30, 40]:
            save = True
            save_path = f'Assignment1/{folder_name}/{title}.png' 


        r = membrane_potential(
            I_var = i0,
            save = save,
            save_path = save_path,
            title = title,
            a=a,
            b=b,
            c=c,
            d=d
        )
        R.append(r)
    
    # np.save('RS.npy', R)
    plot_spike(tspan=range(num_trials), array = R, ylabel='num_spikes/steps',xlabel='I', 
        maxy=None, miny=None, color = 'r', save_path = f'Assignment1/{folder_name}/freq_spikes.png',
        title = 'spike frequency in the last 800 steps', save = True,
        label=folder_name,
        compare_array=compare_array
        )


def main():
    pass

if __name__ == '__main__':
    main()