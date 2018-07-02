'''
© Author: Saunak Saha
Iowa State University
'''
import numpy as np
import pandas as pd
import random
import time
import sys
import os
from brian2 import *
import matplotlib
#matplotlib.use('agg')
from scipy.misc import toimage
from matplotlib import pylab, pyplot
matplotlib.pyplot.switch_backend('agg')
from matplotlib.image import imread
from matplotlib.backends.backend_pdf import PdfPages
prefs.codegen.target = 'numpy'




def trainDataLoad(trainfile):
    trainData = pd.read_csv(trainfile, header=None)
    trainImg = (trainData.iloc[:,1:]).values
    
    return trainImg

def weightLoad(weightfile):
    weights = pd.read_csv(weightfile, header = None)
    weights = weights.values

    return weights

def paramLoad(parameterfile):
    f = open(parameterfile,"r")
    lines = f.readlines()
    params = []
    for x in lines:
        params.append(x.split()[2])
    f.close()
    
    return params

def stimulusGenerator(examples_start, examples_end, rate_divider):
    stimulus = TimedArray(np.vstack([[trainImg[c,:], trainImg[c,:], trainImg[c,:], 
                                trainImg[c,:], trainImg[c,:], trainImg[c,:], 
                                trainImg[c,:], np.zeros(N_Ip), np.zeros(N_Ip),
                               np.zeros(N_Ip)] for c in range(examples_start, examples_end)]), dt=50*ms)


    
    Ip = PoissonGroup(N_Ip, rates = '(stimulus(t,i)/rate_divider)*Hz')
    SpikeMon=SpikeMonitor(Ip)

    SpikeNet= Network(Ip, SpikeMon)
    SpikeNet.run(duration)

    spikes_i=SpikeMon.i
    spikes_t=SpikeMon.t
    
    return (spikes_i, spikes_t)

def NeuronGroups(N_exc, tau_exc, Vth_exc, Vr_exc, Ref_exc,
                N_inh, tau_inh, Vth_inh, Vr_inh, Ref_inh, 
                tau_theta, del_theta, spikes_i, spikes_t):
    
    
    #Input Layer
    SGG= SpikeGeneratorGroup(N_Ip, spikes_i, spikes_t)

    #Excitatory Layer
    eqs_exc = '''
    dv/dt = (-v)/tau_exc: volt (unless refractory)
    dtheta/dt = (-theta)/tau_theta: volt
    '''
    reset = '''
    v= Vr_exc
    theta+=del_theta
    '''
    Exc = NeuronGroup(N_exc, eqs_exc, threshold='v>(Vth_exc+theta)', reset=reset, refractory=Ref_exc, method='euler')
    StateMon_exc = StateMonitor(Exc, ('v','theta'), record=False)
    SpikeMon_exc = SpikeMonitor(Exc)


    #Inhibitory Layer
    eqs_inh = '''
    dv/dt = (-v)/tau_inh: volt (unless refractory)
    '''
    Inh = NeuronGroup(N_inh, eqs_inh, threshold='v>Vth_inh', reset='v=Vr_inh', refractory=Ref_inh, method='euler')
    StateMon_inh = StateMonitor(Inh, 'v', record=False)
    
    return (SGG, Exc, Inh, StateMon_exc, StateMon_inh)

def SynapseGroups(N_exc,taupre, taupost, Apre, Apost, wmax, wmin, nu_pre, nu_post,
                 weights, we2i, wi2e):
    
    #Delays

    minDelay_S1 = 0*ms
    maxDelay_S1 = 10*ms
    delDelay_S1 = maxDelay_S1-minDelay_S1

    minDelay_S2 = 0*ms
    maxDelay_S2 = 5*ms
    delDelay_S2 = maxDelay_S2-minDelay_S2

    minDelay_S3 = 0*ms
    maxDelay_S3 = 1*ms
    delDelay_S3 = maxDelay_S3-minDelay_S3

    #Input-Excitatory

    S1=Synapses(SGG, Exc, '''
             w : volt
             dapre/dt = -apre/taupre : volt (event-driven)
             dapost/dt = -apost/taupost : volt (event-driven)
             ''',
             on_pre='''
             v_post += w
             apre =Apre*mV
             w = clip(w+nu_pre*apost,0,wmax)
             ''',
             on_post='''
             apost =Apost*mV
             w = clip(w+nu_post*apre,0,wmax)
             ''')

    S1.connect(True)
    S1.delay = 'minDelay_S1+rand()*delDelay_S1'
    for neuron in range(0,N_exc):
        S1.w[:,neuron] = (weights[neuron])*volt


    StateMon_S1 = StateMonitor(S1, ['w', 'apre', 'apost'], record=S1[400,90])


    #Excitatory-Inhibitory

    S2 = Synapses(Exc, Inh, 'w : volt', on_pre = 'v_post+=w')
    S2.connect(j ='i')
    S2.w = we2i
    S2.delay = 'minDelay_S2+rand()*delDelay_S2'


    #Inhibtory-Excitatory

    S3 = Synapses(Inh, Exc, 'w : volt', on_pre = 'v_post-=w')
    S3.connect(condition ='j!=i')
    S3.w = wi2e
    S3.delay = 'minDelay_S3+rand()*delDelay_S3'
    
    return (S1, S2, S3, StateMon_S1)

def SSMonitors(StateMonitor_exc, StateMonitor_inh, StateMonitor_S1):
    figure(figsize = (20,20))
    #for m in range(0,1):
    #subplot(N_exc,1,m+1)
    subplot(2,1,1)
    plot(StateMon_exc.t/ms,StateMon_exc.v[56]/mV, 'C1')
    plot(StateMon_inh.t/ms,StateMon_inh.v[56]/mV, 'C2') 
    plot(StateMon_exc.t/ms, StateMon_exc.theta[56]/mV, 'C3')
    subplot(2,1,2)
    plot(StateMon_exc.t/ms,StateMon_exc.v[57]/mV, 'C1')
    plot(StateMon_inh.t/ms,StateMon_inh.v[57]/mV, 'C2') 
    plot(StateMon_exc.t/ms, StateMon_exc.theta[57]/mV, 'C3')
    
    return

def RFMonitor(N_exc,S1,cmap):
    grid = np.zeros((N_exc,28,28))
    for x in range(0,N_exc):
    	grid[x] = np.array(S1.w[:,x].reshape(28,28))



    fig1, axes = pyplot.subplots(nrows=10, ncols=10, sharex = 'col', sharey = 'row', figsize=(8,8))

    for x in range(0,10):
        for y in range(0,10):
             im = axes[x,y].imshow(grid[(x*10)+y], interpolation ='none', aspect = 'auto', cmap=cmap)

    fig1.colorbar(im, ax = axes.ravel().tolist())
    return fig1

def SaveWeights(N_exc):
    trainedWeights=np.zeros((N_exc,784))
    for m in range(0,N_exc):
    	trainedWeights[m]=np.array(S1.w[:,m])
    if os.path.exists('trainedWeights.csv'):
        os.remove('trainedWeights.csv')
    np.savetxt('trainedWeights.csv', trainedWeights, delimiter=",")
    
    return

#-----------------------------------------------------------------
# Data Loading and Processing 
#-----------------------------------------------------------------

start_time = time.time()

trainImg = trainDataLoad('mnist_train.csv')

print("--- Data Load Time %s (s) ---" % (time.time() - start_time))

start_time_1 = time.time()

examples_start = int(sys.argv[1])
examples_end = int(sys.argv[2])

num_examples = (examples_end - examples_start)
duration = (350+150)*num_examples*ms

sim_Mode = sys.argv[3]
if (sim_Mode == "True"):
    paramName = sys.argv[4]
    simNumber = int(sys.argv[5])
    print("--- Entering Simulation Mode ---")

else: 
    simNumber = 1
    print("--- Standalone Mode ---")

N_Ip=784

#-----------------------------------------------------------------
# Simulation starts here
#-----------------------------------------------------------------

start_time = time.time()

if os.path.exists("RF_results.pdf"):
  os.remove("RF_results.pdf")
figfile = PdfPages('RF_results.pdf')


for simCount in range(0,simNumber):
    
    start_time = time.time()
    
    if (sim_Mode == "True"):
        params = paramLoad('parameters_'+paramName+'_'+str(simCount)+'.txt')
    else:
        params = paramLoad('parameters_default.txt')
    
    #-----------------------------------------------------------------
    # Stimulus generation  
    #-----------------------------------------------------------------


    rate_divider = float(params[0])
    spikes_i, spikes_t = stimulusGenerator(examples_start, examples_end, rate_divider)
    

    #-----------------------------------------------------------------
    # Neuron Parameters 
    #-----------------------------------------------------------------
    
    #Excitatory

    N_exc = int(params[1])
    tau_exc = float(params[2])*ms
    Vth_exc = float(params[3])*mV
    Vr_exc = float(params[4])*mV
    Ref_exc = float(params[5])*ms

    #Inhibitory 

    N_inh = float(params[6])
    tau_inh = float(params[7])*ms
    Vth_inh = float(params[8])*mV
    Vr_inh = float(params[9])*mV
    Ref_inh = float(params[10])*ms

    #Homeostasis

    tau_theta = float(params[11])*ms
    del_theta = float(params[12])*mV
    
    
    #Build Neuron Groups with parameters


    SGG,Exc,Inh,StateMon_exc, StateMon_inh = NeuronGroups(N_exc, tau_exc, Vth_exc, Vr_exc, Ref_exc,
                N_inh, tau_inh, Vth_inh, Vr_inh, Ref_inh, 
                tau_theta, del_theta, spikes_i, spikes_t)
    
    #-----------------------------------------------------------------
    # Synapse Parameters
    #-----------------------------------------------------------------
    
    
    #STDP

    taupre = float(params[13])*ms
    taupost = float(params[14])*ms
    Apre = float(params[15])
    Apost = float(params[16])
    wmax = float(params[17])*mV                    
    wmin = float(params[18])*mV
    nu_pre = float(params[19])
    nu_post = float(params[20])
    
    #Ip - Excitatory

    weights = np.zeros((N_exc,784))

    if (examples_start != 0):
        if os.path.exists('trainedWeights.csv'):
            weights = weightLoad('trainedWeights.csv')
        else:
            weights.fill(1e-4)
    
    if (examples_start == 0): 
        weights.fill(1e-4)

    #Excitatory - Inhibitory

    we2i = float(params[21])*mV
    wi2e = float(params[22])*mV
    
    #Build Synapse Groups with parameters
    
    S1, S2, S3, StateMon_S1 = SynapseGroups(N_exc, taupre, taupost, Apre, Apost, wmax, wmin, nu_pre, nu_post,
                 weights, we2i, wi2e)
    
    #-----------------------------------------------------------------
    # Build Network
    #-----------------------------------------------------------------
    
    SNNet = Network(SGG,Exc,Inh,
                   S1,S2,S3,
	           StateMon_exc, StateMon_inh, StateMon_S1)
    
    SNNet.run(duration)
    
    #-----------------------------------------------------------------
    # Weights/Receptive-Fields
    #-----------------------------------------------------------------
    
    
    figure = RFMonitor(N_exc,S1,'YlOrRd')
    if(sim_Mode == "True"):
        figure.suptitle(paramName+' = value '+str(simCount))
    if(sim_Mode == "False"):
        figure.suptitle('All default values')
    figfile.savefig(figure)
    

    #-----------------------------------------------------------------
    # Saving Trained Weights
    #-----------------------------------------------------------------

    SaveWeights(N_exc)
    if(sim_Mode == "True"):
        os.rename('trainedWeights.csv', 'trainedWeights_'+paramName+'_'+str(simCount)+'.csv')

    if(sim_Mode == "True"):
        print("--- Simulation %s completed in %s (s) ---" % (simCount, (time.time() - start_time)))


figfile.close()
print("--- Total exec time: %s (s) ---" % (time.time() - start_time_1))