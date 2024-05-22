###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import sys
if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle
import os

class Modelmanager:

    def __init__(self, file, uploader):
        
        self.file = file            #Datafile used for pickle
        self.uploader = uploader    #Function to upload models                  

        if(os.path.isfile(self.file)):
            self.__load()
        else:
            self.models = []
            self.numModels = 0

    def __load(self):

        # Open the file in binary mode 
        with open(self.file, 'rb') as pfile: 
      
            # Call load method to deserialze 
            pickledata = pickle.load(pfile)
            self.models =       pickledata[0]
            self.numModels =    pickledata[1]

    def __save(self):

        #Save pickled data
        with open(self.file, 'wb') as pfile: 
      
            # A new file will be created 
            pickle.dump((self.models, self.numModels), pfile)

    def __upload(self):
        list = []
        for entry in self.models:
            if(not self.uploader(entry)):
               list.append(entry)
        self.models = list

    def __call__(self, data):
        self.models.append(data)    #Append data
        self.numModels+=1           #Increment counter
        self.__upload()             #Upload models in list
        self.__save()               #Save the models that could not be uploaded
        
    def Counts(self):
        #Total, waiting
        return self.numModels,len(self.models)