#Inspired by in the paper by Izhikevich E.M. (2004)

class Neuron:
    def __init__(self, 
        neuron_type, 
        starting_time = 0,
        starting_potential = -65,
        beginning_delay = 0,
        discrete_time_interval = 0.25 #s
        ):
        self.V = starting_potential
        if neuron_type == 'CH':
            self.a=0.02; self.b=0.2; self.c=-50;  self.d=2
        elif neuron_type == 'RS':
            self.a=0.02; self.b=0.25; self.c=-65;  self.d=6
        elif neuron_type == 'FS':
            self.a=0.1; self.b=0.2; self.c=-65;  self.d=2

        self.VV = []
        self.u = self.b*self.V
        self.uu = []
        self.tau = discrete_time_interval

        self.T1=beginning_delay                            #T1 is the time at which the step input rises
        self.spike_ts = []
        self.cur_t = starting_time

    def spike(self, cur_t, I_var):
        if cur_t>self.T1:
            I = I_var
        else:
            I = 0

        self.V += self.tau*(0.04*self.V**2+5*self.V+140-self.u+I)
        self.u += self.tau*self.a*(self.b*self.V-self.u)

        if self.V > 30:              #If this is a spike
            self.VV.append(30)       #VV is the time-series of membrane potentials
            self.V = self.c
            self.u += self.d
            self.spike_ts.append(1)  #Record a spike
        else:
            self.VV.append(self.V)
            self.spike_ts.append(0)  #Record a spike

        self.uu.append(self.u)
   
    def get_v(self):
        #Get the latest potential, before it was assign c
        return self.VV[-1]
        
    def get_vv(self):
        return self.VV
    
    def get_spike_ts(self):
        return self.spike_ts

def main():
    pass

if __name__ == '__main__':
    main()