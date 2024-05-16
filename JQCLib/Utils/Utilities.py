import numpy as np
unarycount = 4

def fromTextArray(circuit, wash = True):
    
    unarygates = {

        '': 0,
        'i': 0,
        'h': 1,                
        'rx': 2,
        'ry': 3,
        'rz': 4           
    }

    def cx(s,t):    #Source and target of CX        
        assert s != t, "Not a valid circuit"
        assert 0 <= t <= unarycount, "Not a valid circuit"
        
        return t + unarycount + 1 - (1 if s < t else 0)

    def toGate(sg,idx):
        if(sg[0:2].lower() == 'cx'):
            return cx(idx,int(sg[3]))
        else:
            nonlocal unarygates
            return unarygates[sg.lower()]

    data = []
    for idx, l in enumerate(circuit):
        data.append([toGate(g,idx) for g in l])

    if(wash):
        return washCircuit(np.array(data))
    else:
        return np.array(data)

#Remove 'illegal' gates in a column
def detangle(l):

    #Create dictionary for collisions
    collisions = dict(zip(range(len(l)), [[] for i in range(len(l))]))

    #CX gates positions
    source = np.where(l > unarycount)

    #iterate CX 'source','targets' and populate collisions
    for s,t in zip(source[0],(l[source]-(unarycount+1))):
        collisions[t if t < s else t+1].append(s)    

    #Remove collisions
    for t, source in collisions.items(): 
        while(len(source) > 1):
            l[source.pop(-1)] = 0     

        #Target qubit of CX gate should be I
        if(len(source) == 1):
            l[t] = 0

    return l

def washCircuit(data):

    layers = data.shape[1]

    #Get rid of CX gates that don't fit within a single layer
    for i in range(layers):
        data[:,i] = detangle(data[:,i])

    return data

def fromRaw(data):

    #Convert to np.array if list
    if(isinstance(data, list)):
        data = np.array(data)
    
    #Check type of np.array
    if(isinstance(data.dtype, type(np.dtype(int)))):            
        data = washCircuit(data)
    elif(isinstance(data.dtype, type(np.dtype(str)))):            
        data = fromTextArray(data)
    else:
        raise Exception("Circuit array must be string or integer")        

    return data