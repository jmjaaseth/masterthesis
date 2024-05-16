import numpy as np

def oneHot(v,l):
    oh = [0]*l
    oh[v] = 1
    return oh

def crossEntropyLoss(y_pred, y_true):

    loss = 0

    #Convert scalar label to one hot vector
    y_true = oneHot(y_true,len(y_pred))
    
    #Cross entropy loss
    for i in range(len(y_pred)):
        loss += -1 * y_true[i]*np.log(y_pred[i])
    return loss

def probabilityMapper(classes):  

    def mapper(distribution, classes=classes):

        newdist = [0]*classes
        for idx,d in enumerate(distribution):
            newdist[idx%classes] += d
        return np.array(newdist)
    
    return mapper

def modelAccuracy(y_hat, y_true):
    return (y_hat == y_true).sum()/y_hat.shape[0]

def getTerminationCriterion(PATIENCE, MAXIMIZE):
    
    def doTerminate(val_cel, PATIENCE = PATIENCE, MAXIMIZE = MAXIMIZE):   

        if(len(val_cel) >= PATIENCE):
            slope = np.polyfit(range(PATIENCE), val_cel[-PATIENCE:], 1)[0]            
            if(MAXIMIZE):
                return slope < 0
            return slope > 0
        return False
        
    return doTerminate