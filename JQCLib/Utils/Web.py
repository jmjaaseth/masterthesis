###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import sys
if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle
import requests
import urllib.parse
import base64
from progressbar import ProgressBar, Bar, Percentage

 
def Webcall(r):

    def retWrapper(retObj, success = False):
        return success, retObj
    
    #Connect
    try:        
        r.raise_for_status()        
    except requests.exceptions.HTTPError as errh:
        return retWrapper(errh.args[0])
    except requests.exceptions.ReadTimeout as _:         
        return retWrapper("Time out")
    except requests.exceptions.ConnectionError as _:         
        return retWrapper("Connection error")
    except requests.exceptions.RequestException as _:         
        return retWrapper("Exception request")

    return retWrapper(r.text,True)

def Webget(url, timeout = 1):
    return Webcall(requests.get(url, timeout=timeout, verify=True))

def Webpost(url, data, timeout = 1):
    return Webcall(requests.post(url, data = data, timeout=timeout, verify=True))

def WebEncode(s):

    #Encode base64
    s = base64.standard_b64encode(s)
    
    #Convert to UTF-8 string
    s = str(s,encoding='utf-8')
  
    #Url encode the string
    s = urllib.parse.quote_plus(s,encoding='utf-8')

    return s

def WebDecode(s):

    #Url decode the string
    s = urllib.parse.unquote_plus(s)
    
    #Convert to byte array
    s = bytes(s,encoding='utf-8')
  
    #Decode base64
    s = base64.standard_b64decode(s)

    return s

def getUploader(dataset, username, url):

    def PostCircuit(object, dataset = dataset, username = username, url = url):

        postdata =  {
            'dataset'     : dataset,
            'contributor': username,
            'circuitdata': WebEncode(pickle.dumps(object))
        }            
        success, response = Webpost(url,postdata)
        
        #Log here if needed
        #print(f"Success: {success}, Response: {response}")

        return success

    return PostCircuit