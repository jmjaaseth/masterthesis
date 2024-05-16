from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from JQCLib.Utils.Utilities import washCircuit

import os
import numpy as np

#Make bounds dictionary for Bayesian Optimization Process
def BoundsBuilder(qubits, layers, gates):

    keys = [str(i)+"_"+str(j) for i in range(qubits) for j in range(layers)]
    return dict(zip(keys, [(1,gates)]*qubits*layers))

#Run Bayesian Optimization Process
def BayeOptCircuit(blackbox, qubits, layers, gates, logfile, initialPoints, iterations):

    #Circuit dimensions
    dim = (qubits, layers) 
    
    def toCircuit(**kwargs):

        #Set up np array to hold circuit
        nonlocal dim
        data = np.zeros(dim,dtype=int)

        #Fill array with current values
        for key, value in kwargs.items():            
            key = [int(i) for i in key.split('_')]
            data[key[0],key[1]] = int(value)

        return washCircuit(data)

    #Black Box Wrapper function
    def BBWrapper(**kwargs):       

        #return washed array
        return blackbox(toCircuit(**kwargs))

    #Create optimizer
    baye_opt = BayesianOptimization(
        f=BBWrapper,
        pbounds=BoundsBuilder(qubits,layers,gates),
        verbose=2,
        random_state=1,
        allow_duplicate_points=True   #Kan sikkert fjernes
    )

    #Enable logging
    logger = JSONLogger(path=logfile,reset=False)

    #Load data if log file is present
    if(os.path.isfile(logfile)):
        
        #Drop initial points if loading from file
        initialPoints = 0

        #Load data
        load_logs(baye_opt, logs=[logfile]);
    
    #Subscribe to event to enable log writing
    baye_opt.subscribe(Events.OPTIMIZATION_STEP, logger)    

    #Run process
    baye_opt.maximize(init_points=initialPoints, n_iter=iterations)

    #Return max value & circuit data
    return baye_opt.max['target'], toCircuit(**baye_opt.max['params']), baye_opt.res