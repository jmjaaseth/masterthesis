###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import sys
if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle
import os
if os.name == 'nt':
    import importlib.resources as importlib_resources
else:
    import importlib_resources

def LoadFixedIris():
    return AsPackage()

def AsPackage():  

    bytedata = importlib_resources.read_binary("JQCLib.Data", "IrisFixed.pkl")
    r = pickle.loads(bytedata)
    
    return r[0],r[1],r[2],r[3],r[4],r[5]

def AsFile():

    with open('JQCLib/Data/IrisFixed.pkl', 'rb') as infile:
        r = pickle.load(infile)
    
    return r[0],r[1],r[2],r[3],r[4],r[5]

def LoadLooseIris():

    bytedata = importlib_resources.read_binary("JQCLib.Data", "IrisLoose.pkl")    
    return pickle.loads(bytedata)
