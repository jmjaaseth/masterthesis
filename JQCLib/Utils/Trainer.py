###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

from JQCLib.Classes.Optimizers import Adam
from JQCLib.Classes.jQNNClass import jQNN
from JQCLib.Classes.JMStateVectorSimClass import JMStateVectorSim, FeatureEncode
from JQCLib.Classes.TrainingResultClass import TrainingResult

from JQCLib.Utils.MLUtils import crossEntropyLoss, probabilityMapper, modelAccuracy, getTerminationCriterion
from JQCLib.Utils.Dataloaders import LoadFixedIris

#Some other libs
import numpy as np
import time

#Fixed parameters
#BATCH_SIZE =    25 #Produces semi-smooth learning curves
BATCH_SIZE =    120 #Minimize variation on results setting BATCH_SIZE = Dataset Size
PERTUBATION =   .2  #Found by trial/error
PATIENCE =      20  #Found by trial/error
LEARNINGRATE =  .5  #Found by trial/error

#declare variables
X_train, y_train, X_val, y_val, X_tst, y_tst = LoadFixedIris()

#declare functions
validationscore = testacc = testce = terminate = opt = None

#Initialization function
def SetUpEnv(m):

    global X_train, y_train, X_val, y_val, X_tst, y_tst, model

    model = m

    global validationscore, testacc, testce, terminate, opt

    #Score functions
    validationscore =   lambda weights : modelAccuracy(np.array([model.predict(x,weights) for x in X_val]),y_val)
    testacc =           lambda weights : modelAccuracy(np.array([model.predict(x,weights) for x in X_tst]),y_tst)
    testce =            lambda weights : np.mean([crossEntropyLoss(model.forward(X_tst[idx],weights),y_tst[idx]) for idx in range(len(y_tst))])

    #Optimizer & termination criterion
    terminate =         getTerminationCriterion(MAXIMIZE = False, PATIENCE = PATIENCE)
    opt =               Adam(model.weights,eta=LEARNINGRATE)

def calcGradients(state, weights, y):
    
    grads = []
    for idx in range(len(weights)): 

        wp = weights.copy()
        wm = weights.copy()

        wp[idx]+=PERTUBATION
        wm[idx]-=PERTUBATION 
        
        lp = crossEntropyLoss(model.forward(state,wp),y)
        lm = crossEntropyLoss(model.forward(state,wm),y)        
        grads.append((lp-lm)/(2*PERTUBATION))
        
    return np.array(grads)

def batchloss(weights):

    #Pick indices for this batch
    indices = np.random.choice(len(X_train), BATCH_SIZE, False)

    #Total loss
    loss = [crossEntropyLoss(model.forward(X_train[idx],weights),y_train[idx]) for idx in indices]    

    #Total gradients
    totgrads = np.array([calcGradients(X_train[idx], weights, y_train[idx]) for idx in indices])    

    #Average gradients
    avgGrads = np.mean(totgrads,axis=0,keepdims=True)
    return np.mean(loss), avgGrads.flatten()

def descent(model):   

    #Init environment
    SetUpEnv(model)

    #Keep track of weights for early stopping
    wl = []

    #Return object
    tr = TrainingResult(model.CircuitData())    

    #Time the descent
    st = time.time()

    #Mini-batch SGD
    while not terminate(tr.validationLoss):

        weights = opt.getWeights()              #Get weights from optimizer
        trainloss, grads = batchloss(weights)   #Calc loss on training data and gradients        
        opt.step(grads)                         #Step optimizer        
        wl.append(opt.getWeights())             #Keep track of weights

        #calculate validation loss
        valloss = np.mean([crossEntropyLoss(model.forward(X_val[idx],weights),y_val[idx]) for idx in range(len(y_val))])
        
        #report data to return object
        tr.report(trainloss,valloss,False)
        
    #After training
    #----------------------------------------------------------------------------

    #Set total execution time
    tr.execTime(time.time() - st)

    #Find best weights using lowest cross entropy on validation data
    bestWeights = wl[tr.bestIndex()]

    #Score model on test data
    tr.setScores(testacc(bestWeights), testce(bestWeights))
    
    return tr

#Run everything
#----------------------------------------------------------------------------------------------

def SampleAndTrain(Qubits = 4, Layers = 12):
    return Train(np.random.randint(0,8,(Qubits,Layers)))

def Train(cd):     
     return descent(jQNN(JMStateVectorSim.FromArray(cd),probabilityMapper(3)))