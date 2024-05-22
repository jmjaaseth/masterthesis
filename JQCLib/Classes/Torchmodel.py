###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import torch.nn.functional as F
import torch
import torch.nn as nn

class ConfigurableCNN(nn.Module):    
    
    hardCodedchkpntFile = "JQCLib\\Results\\CNN_chkpnt.pth"   

    def LoadBest():
        if(torch.cuda.is_available()):
            checkpoint = torch.load(ConfigurableCNN.hardCodedchkpntFile)
        else:
            checkpoint = torch.load(ConfigurableCNN.hardCodedchkpntFile,map_location=torch.device('cpu'))

        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        return model

    def __init__(self, chnls1,chnls2, l1, l2, l3, d1, d2, d3, d4):
        super().__init__()

        #Force ints
        chnls1 = int(chnls1)
        chnls2 = int(chnls2)

        l1 = int(l1)
        l2 = int(l2)
        l3 = int(l3)

        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4

        self.conv1 = nn.Conv2d(in_channels=1,       out_channels=chnls1,  kernel_size=(1,2),  stride=1,   padding=0)
        self.conv2 = nn.Conv2d(in_channels=chnls1,  out_channels=chnls2,  kernel_size=(2,1),  stride=1,   padding=0)

        #Dim after conv is Batch x chnls2 x 3 x 11
        self.fc1 = nn.Linear(chnls2 * 3 * 11, l1)        
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc4 = nn.Linear(l3, 1)

    def Save(self):
        
        checkpoint = {
            'model': self,
            'state_dict': self.state_dict()    
        }

        torch.save(checkpoint, ConfigurableCNN.hardCodedchkpntFile)

    def forward(self, x):

        #CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        #Flatten
        x = torch.flatten(x, 1)

        #Alternating FC & Dropout 
        x = F.dropout(x,self.d1,self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,self.d2,self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,self.d3,self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x,self.d4,self.training)

        return self.fc4(x)
    

class MLP(nn.Module):    
    
    hardCodedchkpntFile = "JQCLib\\Results\\MLP_chkpnt.pth"   

    def LoadBest():
        if(torch.cuda.is_available()):
            checkpoint = torch.load(MLP.hardCodedchkpntFile)
        else:
            checkpoint = torch.load(MLP.hardCodedchkpntFile,map_location=torch.device('cpu'))

        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        return model

    def __init__(self):
        super().__init__()


        chnls = 20
        self.conv1 = nn.Conv2d(in_channels=1,       out_channels=chnls,  kernel_size=(1,2),  stride=1,   padding=0)
        
        self.fc1 = nn.Linear(chnls * 4 * 11, 512)        
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def Save(self):
        
        checkpoint = {
            'model': self,
            'state_dict': self.state_dict()    
        }

        torch.save(checkpoint, MLP.hardCodedchkpntFile)

    def forward(self, x):   

        x = F.relu(self.conv1(x))

        #Flatten
        x = torch.flatten(x, 1)     
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x,.2,self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)