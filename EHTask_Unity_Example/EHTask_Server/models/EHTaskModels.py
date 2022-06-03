# Copyright (c) Hu Zhiming 2021/04/22 jimmyhu@pku.edu.cn All Rights Reserved.

import torch
import torch.nn as nn
from math import floor

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EHTask(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Head feature size: {}'.format(self.headFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_CNN1D Module
        eyeCNN1D_outChannels1 = 16
        eyeCNN1D_kernelSize1 = 3
        eyeCNN1D_featureSize1 = floor((self.eyeFeatureLength - eyeCNN1D_kernelSize1 + 1)/2)
        eyeCNN1D_outChannels2 = 16
        eyeCNN1D_kernelSize2 = 3
        eyeCNN1D_featureSize2 = floor((eyeCNN1D_featureSize1 - eyeCNN1D_kernelSize2 + 1)/2)
        eyeCNN1D_outChannels3 = 16
        eyeCNN1D_kernelSize3 = 3
        eyeCNN1D_featureSize3 = floor((eyeCNN1D_featureSize2 - eyeCNN1D_kernelSize3 + 1)/2)
        self.eyeCNN1D_outputSize = eyeCNN1D_featureSize3 * eyeCNN1D_outChannels3
        self.eyeCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.eyeFeatureNum, out_channels=eyeCNN1D_outChannels1,kernel_size=eyeCNN1D_kernelSize1),
            nn.BatchNorm1d(eyeCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=eyeCNN1D_outChannels1, out_channels=eyeCNN1D_outChannels2,kernel_size=eyeCNN1D_kernelSize2),
            nn.BatchNorm1d(eyeCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=eyeCNN1D_outChannels2, out_channels=eyeCNN1D_outChannels3,kernel_size=eyeCNN1D_kernelSize3),
            nn.BatchNorm1d(eyeCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
        
        # Eye_GRU Module
        self.eyeGRU_hidden_size = 64
        self.eyeGRU_layers = 1
        self.eyeGRU_directions = 2
        self.eyeGRU = nn.GRU(eyeCNN1D_outChannels3,self.eyeGRU_hidden_size, self.eyeGRU_layers, batch_first=True, bidirectional=bool(self.eyeGRU_directions-1))
        
        # Head_CNN1D Module
        headCNN1D_outChannels1 = 16
        headCNN1D_kernelSize1 = 3
        headCNN1D_featureSize1 = floor((self.headFeatureLength - headCNN1D_kernelSize1 + 1)/2)
        headCNN1D_outChannels2 = 16
        headCNN1D_kernelSize2 = 3
        headCNN1D_featureSize2 = floor((headCNN1D_featureSize1 - headCNN1D_kernelSize2 + 1)/2)
        headCNN1D_outChannels3 = 16
        headCNN1D_kernelSize3 = 3
        headCNN1D_featureSize3 = floor((headCNN1D_featureSize2 - headCNN1D_kernelSize3 + 1)/2)
        self.headCNN1D_outputSize = headCNN1D_featureSize3 * headCNN1D_outChannels3
        self.headCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headCNN1D_outChannels1,kernel_size=headCNN1D_kernelSize1),
            nn.BatchNorm1d(headCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=headCNN1D_outChannels1, out_channels=headCNN1D_outChannels2,kernel_size=headCNN1D_kernelSize2),
            nn.BatchNorm1d(headCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=headCNN1D_outChannels2, out_channels=headCNN1D_outChannels3,kernel_size=headCNN1D_kernelSize3),
            nn.BatchNorm1d(headCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),       
             )
        
        # Head_GRU Module
        self.headGRU_hidden_size = 64
        self.headGRU_layers = 1
        self.headGRU_directions = 2
        self.headGRU = nn.GRU(headCNN1D_outChannels3,self.headGRU_hidden_size, self.headGRU_layers, batch_first=True, bidirectional=bool(self.headGRU_directions-1))
        
        # GW_CNN1D Module
        gwCNN1D_outChannels1 = 16
        gwCNN1D_kernelSize1 = 3
        gwCNN1D_featureSize1 = floor((self.gwFeatureLength - gwCNN1D_kernelSize1 + 1)/2)
        gwCNN1D_outChannels2 = 16
        gwCNN1D_kernelSize2 = 3
        gwCNN1D_featureSize2 = floor((gwCNN1D_featureSize1 - gwCNN1D_kernelSize2 + 1)/2)
        gwCNN1D_outChannels3 = 16
        gwCNN1D_kernelSize3 = 3
        gwCNN1D_featureSize3 = floor((gwCNN1D_featureSize2 - gwCNN1D_kernelSize3 + 1)/2)
        self.gwCNN1D_outputSize = gwCNN1D_featureSize3 * gwCNN1D_outChannels3
        self.gwCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gwFeatureNum, out_channels=gwCNN1D_outChannels1,kernel_size=gwCNN1D_kernelSize1),
            nn.BatchNorm1d(gwCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=gwCNN1D_outChannels1, out_channels=gwCNN1D_outChannels2,kernel_size=gwCNN1D_kernelSize2),
            nn.BatchNorm1d(gwCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=gwCNN1D_outChannels2, out_channels=gwCNN1D_outChannels3,kernel_size=gwCNN1D_kernelSize3),
            nn.BatchNorm1d(gwCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),
             )
        
        # GW_GRU Module
        self.gwGRU_hidden_size = 64
        self.gwGRU_layers = 1
        self.gwGRU_directions = 2
        self.gwGRU = nn.GRU(gwCNN1D_outChannels3,self.gwGRU_hidden_size, self.gwGRU_layers, batch_first=True, bidirectional=bool(self.gwGRU_directions-1))
        
        # task prediction FC Module
        eyeGRU_output_size = self.eyeGRU_hidden_size*self.eyeGRU_directions
        headGRU_output_size = self.headGRU_hidden_size*self.headGRU_directions
        gwGRU_output_size = self.gwGRU_hidden_size*self.gwGRU_directions
        prdFC_inputSize = eyeGRU_output_size + headGRU_output_size + gwGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        headFeature = x[:, index: index+self.headFeatureSize]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)
        eyeFeature = eyeFeature.permute(0,2,1)        
        eyeFeatureOut = self.eyeCNN1D(eyeFeature)
        eyeFeatureOut = eyeFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.eyeGRU_layers*self.eyeGRU_directions, x.size(0), self.eyeGRU_hidden_size).to(device) 
        eyeGruOut, _ = self.eyeGRU(eyeFeatureOut, h0)  
        eyeGruOut = eyeGruOut[:, -1, :]
        
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        headFeature = headFeature.permute(0,2,1)        
        headFeatureOut = self.headCNN1D(headFeature)
        headFeatureOut = headFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.headGRU_layers*self.headGRU_directions, x.size(0), self.headGRU_hidden_size).to(device) 
        headGruOut, _ = self.headGRU(headFeatureOut, h0)  
        headGruOut = headGruOut[:, -1, :]

        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        gwFeature = gwFeature.permute(0,2,1)        
        gwFeatureOut = self.gwCNN1D(gwFeature)
        gwFeatureOut = gwFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.gwGRU_layers*self.gwGRU_directions, x.size(0), self.gwGRU_hidden_size).to(device) 
        gwGruOut, _ = self.gwGRU(gwFeatureOut, h0)  
        gwGruOut = gwGruOut[:, -1, :]
        
        out = torch.cat((eyeGruOut, headGruOut), 1)
        out = torch.cat((out, gwGruOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out
    
    
# EHTask model using only eye-in-head features.    
class EHTask_Eye(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize        
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        
        # Eye_CNN1D Module
        eyeCNN1D_outChannels1 = 16
        eyeCNN1D_kernelSize1 = 3
        eyeCNN1D_featureSize1 = floor((self.eyeFeatureLength - eyeCNN1D_kernelSize1 + 1)/2)
        eyeCNN1D_outChannels2 = 16
        eyeCNN1D_kernelSize2 = 3
        eyeCNN1D_featureSize2 = floor((eyeCNN1D_featureSize1 - eyeCNN1D_kernelSize2 + 1)/2)
        eyeCNN1D_outChannels3 = 16
        eyeCNN1D_kernelSize3 = 3
        eyeCNN1D_featureSize3 = floor((eyeCNN1D_featureSize2 - eyeCNN1D_kernelSize3 + 1)/2)
        self.eyeCNN1D_outputSize = eyeCNN1D_featureSize3 * eyeCNN1D_outChannels3
        self.eyeCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.eyeFeatureNum, out_channels=eyeCNN1D_outChannels1,kernel_size=eyeCNN1D_kernelSize1),
            nn.BatchNorm1d(eyeCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=eyeCNN1D_outChannels1, out_channels=eyeCNN1D_outChannels2,kernel_size=eyeCNN1D_kernelSize2),
            nn.BatchNorm1d(eyeCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=eyeCNN1D_outChannels2, out_channels=eyeCNN1D_outChannels3,kernel_size=eyeCNN1D_kernelSize3),
            nn.BatchNorm1d(eyeCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
        
        # Eye_GRU Module
        self.eyeGRU_hidden_size = 64
        self.eyeGRU_layers = 1
        self.eyeGRU_directions = 2
        self.eyeGRU = nn.GRU(eyeCNN1D_outChannels3,self.eyeGRU_hidden_size, self.eyeGRU_layers, batch_first=True, bidirectional=bool(self.eyeGRU_directions-1))
        
        
        # task prediction FC Module
        eyeGRU_output_size = self.eyeGRU_hidden_size*self.eyeGRU_directions
        prdFC_inputSize = eyeGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)
        eyeFeature = eyeFeature.permute(0,2,1)        
        eyeFeatureOut = self.eyeCNN1D(eyeFeature)
        eyeFeatureOut = eyeFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.eyeGRU_layers*self.eyeGRU_directions, x.size(0), self.eyeGRU_hidden_size).to(device) 
        eyeGruOut, _ = self.eyeGRU(eyeFeatureOut, h0)  
        eyeGruOut = eyeGruOut[:, -1, :]        
        
        out = eyeGruOut
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out   
    
    
# EHTask model using only head features.
class EHTask_Head(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize        
        print('Head feature size: {}'.format(self.headFeatureSize))
        
        # preset params
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        
        # Head_CNN1D Module
        headCNN1D_outChannels1 = 16
        headCNN1D_kernelSize1 = 3
        headCNN1D_featureSize1 = floor((self.headFeatureLength - headCNN1D_kernelSize1 + 1)/2)
        headCNN1D_outChannels2 = 16
        headCNN1D_kernelSize2 = 3
        headCNN1D_featureSize2 = floor((headCNN1D_featureSize1 - headCNN1D_kernelSize2 + 1)/2)
        headCNN1D_outChannels3 = 16
        headCNN1D_kernelSize3 = 3
        headCNN1D_featureSize3 = floor((headCNN1D_featureSize2 - headCNN1D_kernelSize3 + 1)/2)
        self.headCNN1D_outputSize = headCNN1D_featureSize3 * headCNN1D_outChannels3
        self.headCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headCNN1D_outChannels1,kernel_size=headCNN1D_kernelSize1),
            nn.BatchNorm1d(headCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=headCNN1D_outChannels1, out_channels=headCNN1D_outChannels2,kernel_size=headCNN1D_kernelSize2),
            nn.BatchNorm1d(headCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=headCNN1D_outChannels2, out_channels=headCNN1D_outChannels3,kernel_size=headCNN1D_kernelSize3),
            nn.BatchNorm1d(headCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),       
             )
        
        # Head_GRU Module
        self.headGRU_hidden_size = 64
        self.headGRU_layers = 1
        self.headGRU_directions = 2
        self.headGRU = nn.GRU(headCNN1D_outChannels3,self.headGRU_hidden_size, self.headGRU_layers, batch_first=True, bidirectional=bool(self.headGRU_directions-1))
        
        # task prediction FC Module
        headGRU_output_size = self.headGRU_hidden_size*self.headGRU_directions
        prdFC_inputSize = headGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        headFeature = x[:, index: index+self.headFeatureSize]
                
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        headFeature = headFeature.permute(0,2,1)        
        headFeatureOut = self.headCNN1D(headFeature)
        headFeatureOut = headFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.headGRU_layers*self.headGRU_directions, x.size(0), self.headGRU_hidden_size).to(device) 
        headGruOut, _ = self.headGRU(headFeatureOut, h0)  
        headGruOut = headGruOut[:, -1, :]

        out = headGruOut
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out    
    

# EHTask model using only gaze-in-world features.    
class EHTask_GW(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
                
        # GW_CNN1D Module
        gwCNN1D_outChannels1 = 16
        gwCNN1D_kernelSize1 = 3
        gwCNN1D_featureSize1 = floor((self.gwFeatureLength - gwCNN1D_kernelSize1 + 1)/2)
        gwCNN1D_outChannels2 = 16
        gwCNN1D_kernelSize2 = 3
        gwCNN1D_featureSize2 = floor((gwCNN1D_featureSize1 - gwCNN1D_kernelSize2 + 1)/2)
        gwCNN1D_outChannels3 = 16
        gwCNN1D_kernelSize3 = 3
        gwCNN1D_featureSize3 = floor((gwCNN1D_featureSize2 - gwCNN1D_kernelSize3 + 1)/2)
        self.gwCNN1D_outputSize = gwCNN1D_featureSize3 * gwCNN1D_outChannels3
        self.gwCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gwFeatureNum, out_channels=gwCNN1D_outChannels1,kernel_size=gwCNN1D_kernelSize1),
            nn.BatchNorm1d(gwCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=gwCNN1D_outChannels1, out_channels=gwCNN1D_outChannels2,kernel_size=gwCNN1D_kernelSize2),
            nn.BatchNorm1d(gwCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=gwCNN1D_outChannels2, out_channels=gwCNN1D_outChannels3,kernel_size=gwCNN1D_kernelSize3),
            nn.BatchNorm1d(gwCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),
             )
        
        # GW_GRU Module
        self.gwGRU_hidden_size = 64
        self.gwGRU_layers = 1
        self.gwGRU_directions = 2
        self.gwGRU = nn.GRU(gwCNN1D_outChannels3,self.gwGRU_hidden_size, self.gwGRU_layers, batch_first=True, bidirectional=bool(self.gwGRU_directions-1))
        
        # task prediction FC Module
        gwGRU_output_size = self.gwGRU_hidden_size*self.gwGRU_directions
        prdFC_inputSize = gwGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        gwFeature = gwFeature.permute(0,2,1)        
        gwFeatureOut = self.gwCNN1D(gwFeature)
        gwFeatureOut = gwFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.gwGRU_layers*self.gwGRU_directions, x.size(0), self.gwGRU_hidden_size).to(device) 
        gwGruOut, _ = self.gwGRU(gwFeatureOut, h0)  
        gwGruOut = gwGruOut[:, -1, :]
        
        out = gwGruOut
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out    
    

# EHTask model using eye-in-head features and head features.
class EHTask_EyeHead(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Head feature size: {}'.format(self.headFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        
        # Eye_CNN1D Module
        eyeCNN1D_outChannels1 = 16
        eyeCNN1D_kernelSize1 = 3
        eyeCNN1D_featureSize1 = floor((self.eyeFeatureLength - eyeCNN1D_kernelSize1 + 1)/2)
        eyeCNN1D_outChannels2 = 16
        eyeCNN1D_kernelSize2 = 3
        eyeCNN1D_featureSize2 = floor((eyeCNN1D_featureSize1 - eyeCNN1D_kernelSize2 + 1)/2)
        eyeCNN1D_outChannels3 = 16
        eyeCNN1D_kernelSize3 = 3
        eyeCNN1D_featureSize3 = floor((eyeCNN1D_featureSize2 - eyeCNN1D_kernelSize3 + 1)/2)
        self.eyeCNN1D_outputSize = eyeCNN1D_featureSize3 * eyeCNN1D_outChannels3
        self.eyeCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.eyeFeatureNum, out_channels=eyeCNN1D_outChannels1,kernel_size=eyeCNN1D_kernelSize1),
            nn.BatchNorm1d(eyeCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=eyeCNN1D_outChannels1, out_channels=eyeCNN1D_outChannels2,kernel_size=eyeCNN1D_kernelSize2),
            nn.BatchNorm1d(eyeCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=eyeCNN1D_outChannels2, out_channels=eyeCNN1D_outChannels3,kernel_size=eyeCNN1D_kernelSize3),
            nn.BatchNorm1d(eyeCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
        
        # Eye_GRU Module
        self.eyeGRU_hidden_size = 64
        self.eyeGRU_layers = 1
        self.eyeGRU_directions = 2
        self.eyeGRU = nn.GRU(eyeCNN1D_outChannels3,self.eyeGRU_hidden_size, self.eyeGRU_layers, batch_first=True, bidirectional=bool(self.eyeGRU_directions-1))
        
        # Head_CNN1D Module
        headCNN1D_outChannels1 = 16
        headCNN1D_kernelSize1 = 3
        headCNN1D_featureSize1 = floor((self.headFeatureLength - headCNN1D_kernelSize1 + 1)/2)
        headCNN1D_outChannels2 = 16
        headCNN1D_kernelSize2 = 3
        headCNN1D_featureSize2 = floor((headCNN1D_featureSize1 - headCNN1D_kernelSize2 + 1)/2)
        headCNN1D_outChannels3 = 16
        headCNN1D_kernelSize3 = 3
        headCNN1D_featureSize3 = floor((headCNN1D_featureSize2 - headCNN1D_kernelSize3 + 1)/2)
        self.headCNN1D_outputSize = headCNN1D_featureSize3 * headCNN1D_outChannels3
        self.headCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headCNN1D_outChannels1,kernel_size=headCNN1D_kernelSize1),
            nn.BatchNorm1d(headCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=headCNN1D_outChannels1, out_channels=headCNN1D_outChannels2,kernel_size=headCNN1D_kernelSize2),
            nn.BatchNorm1d(headCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=headCNN1D_outChannels2, out_channels=headCNN1D_outChannels3,kernel_size=headCNN1D_kernelSize3),
            nn.BatchNorm1d(headCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),       
             )
        
        # Head_GRU Module
        self.headGRU_hidden_size = 64
        self.headGRU_layers = 1
        self.headGRU_directions = 2
        self.headGRU = nn.GRU(headCNN1D_outChannels3,self.headGRU_hidden_size, self.headGRU_layers, batch_first=True, bidirectional=bool(self.headGRU_directions-1))
                
        # task prediction FC Module
        eyeGRU_output_size = self.eyeGRU_hidden_size*self.eyeGRU_directions
        headGRU_output_size = self.headGRU_hidden_size*self.headGRU_directions
        prdFC_inputSize = eyeGRU_output_size + headGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        headFeature = x[:, index: index+self.headFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)
        eyeFeature = eyeFeature.permute(0,2,1)        
        eyeFeatureOut = self.eyeCNN1D(eyeFeature)
        eyeFeatureOut = eyeFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.eyeGRU_layers*self.eyeGRU_directions, x.size(0), self.eyeGRU_hidden_size).to(device) 
        eyeGruOut, _ = self.eyeGRU(eyeFeatureOut, h0)  
        eyeGruOut = eyeGruOut[:, -1, :]
        
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        headFeature = headFeature.permute(0,2,1)        
        headFeatureOut = self.headCNN1D(headFeature)
        headFeatureOut = headFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.headGRU_layers*self.headGRU_directions, x.size(0), self.headGRU_hidden_size).to(device) 
        headGruOut, _ = self.headGRU(headFeatureOut, h0)  
        headGruOut = headGruOut[:, -1, :]
        
        out = torch.cat((eyeGruOut, headGruOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out    
    
    
# EHTask model using eye-in-head features and gaze-in-world features.    
class EHTask_EyeGW(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_CNN1D Module
        eyeCNN1D_outChannels1 = 16
        eyeCNN1D_kernelSize1 = 3
        eyeCNN1D_featureSize1 = floor((self.eyeFeatureLength - eyeCNN1D_kernelSize1 + 1)/2)
        eyeCNN1D_outChannels2 = 16
        eyeCNN1D_kernelSize2 = 3
        eyeCNN1D_featureSize2 = floor((eyeCNN1D_featureSize1 - eyeCNN1D_kernelSize2 + 1)/2)
        eyeCNN1D_outChannels3 = 16
        eyeCNN1D_kernelSize3 = 3
        eyeCNN1D_featureSize3 = floor((eyeCNN1D_featureSize2 - eyeCNN1D_kernelSize3 + 1)/2)
        self.eyeCNN1D_outputSize = eyeCNN1D_featureSize3 * eyeCNN1D_outChannels3
        self.eyeCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.eyeFeatureNum, out_channels=eyeCNN1D_outChannels1,kernel_size=eyeCNN1D_kernelSize1),
            nn.BatchNorm1d(eyeCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=eyeCNN1D_outChannels1, out_channels=eyeCNN1D_outChannels2,kernel_size=eyeCNN1D_kernelSize2),
            nn.BatchNorm1d(eyeCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=eyeCNN1D_outChannels2, out_channels=eyeCNN1D_outChannels3,kernel_size=eyeCNN1D_kernelSize3),
            nn.BatchNorm1d(eyeCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
        
        # Eye_GRU Module
        self.eyeGRU_hidden_size = 64
        self.eyeGRU_layers = 1
        self.eyeGRU_directions = 2
        self.eyeGRU = nn.GRU(eyeCNN1D_outChannels3,self.eyeGRU_hidden_size, self.eyeGRU_layers, batch_first=True, bidirectional=bool(self.eyeGRU_directions-1))
                
        # GW_CNN1D Module
        gwCNN1D_outChannels1 = 16
        gwCNN1D_kernelSize1 = 3
        gwCNN1D_featureSize1 = floor((self.gwFeatureLength - gwCNN1D_kernelSize1 + 1)/2)
        gwCNN1D_outChannels2 = 16
        gwCNN1D_kernelSize2 = 3
        gwCNN1D_featureSize2 = floor((gwCNN1D_featureSize1 - gwCNN1D_kernelSize2 + 1)/2)
        gwCNN1D_outChannels3 = 16
        gwCNN1D_kernelSize3 = 3
        gwCNN1D_featureSize3 = floor((gwCNN1D_featureSize2 - gwCNN1D_kernelSize3 + 1)/2)
        self.gwCNN1D_outputSize = gwCNN1D_featureSize3 * gwCNN1D_outChannels3
        self.gwCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gwFeatureNum, out_channels=gwCNN1D_outChannels1,kernel_size=gwCNN1D_kernelSize1),
            nn.BatchNorm1d(gwCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=gwCNN1D_outChannels1, out_channels=gwCNN1D_outChannels2,kernel_size=gwCNN1D_kernelSize2),
            nn.BatchNorm1d(gwCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=gwCNN1D_outChannels2, out_channels=gwCNN1D_outChannels3,kernel_size=gwCNN1D_kernelSize3),
            nn.BatchNorm1d(gwCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),
             )
        
        # GW_GRU Module
        self.gwGRU_hidden_size = 64
        self.gwGRU_layers = 1
        self.gwGRU_directions = 2
        self.gwGRU = nn.GRU(gwCNN1D_outChannels3,self.gwGRU_hidden_size, self.gwGRU_layers, batch_first=True, bidirectional=bool(self.gwGRU_directions-1))
        
        # task prediction FC Module
        eyeGRU_output_size = self.eyeGRU_hidden_size*self.eyeGRU_directions
        gwGRU_output_size = self.gwGRU_hidden_size*self.gwGRU_directions
        prdFC_inputSize = eyeGRU_output_size + gwGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)
        eyeFeature = eyeFeature.permute(0,2,1)        
        eyeFeatureOut = self.eyeCNN1D(eyeFeature)
        eyeFeatureOut = eyeFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.eyeGRU_layers*self.eyeGRU_directions, x.size(0), self.eyeGRU_hidden_size).to(device) 
        eyeGruOut, _ = self.eyeGRU(eyeFeatureOut, h0)  
        eyeGruOut = eyeGruOut[:, -1, :]
        
        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        gwFeature = gwFeature.permute(0,2,1)        
        gwFeatureOut = self.gwCNN1D(gwFeature)
        gwFeatureOut = gwFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.gwGRU_layers*self.gwGRU_directions, x.size(0), self.gwGRU_hidden_size).to(device) 
        gwGruOut, _ = self.gwGRU(gwFeatureOut, h0)  
        gwGruOut = gwGruOut[:, -1, :]
        
        out = torch.cat((eyeGruOut, gwGruOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out    
    
    
# EHTask model using head features and gaze-in-world features.
class EHTask_HeadGW(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Head feature size: {}'.format(self.headFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
                
        # Head_CNN1D Module
        headCNN1D_outChannels1 = 16
        headCNN1D_kernelSize1 = 3
        headCNN1D_featureSize1 = floor((self.headFeatureLength - headCNN1D_kernelSize1 + 1)/2)
        headCNN1D_outChannels2 = 16
        headCNN1D_kernelSize2 = 3
        headCNN1D_featureSize2 = floor((headCNN1D_featureSize1 - headCNN1D_kernelSize2 + 1)/2)
        headCNN1D_outChannels3 = 16
        headCNN1D_kernelSize3 = 3
        headCNN1D_featureSize3 = floor((headCNN1D_featureSize2 - headCNN1D_kernelSize3 + 1)/2)
        self.headCNN1D_outputSize = headCNN1D_featureSize3 * headCNN1D_outChannels3
        self.headCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headCNN1D_outChannels1,kernel_size=headCNN1D_kernelSize1),
            nn.BatchNorm1d(headCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=headCNN1D_outChannels1, out_channels=headCNN1D_outChannels2,kernel_size=headCNN1D_kernelSize2),
            nn.BatchNorm1d(headCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=headCNN1D_outChannels2, out_channels=headCNN1D_outChannels3,kernel_size=headCNN1D_kernelSize3),
            nn.BatchNorm1d(headCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),       
             )
        
        # Head_GRU Module
        self.headGRU_hidden_size = 64
        self.headGRU_layers = 1
        self.headGRU_directions = 2
        self.headGRU = nn.GRU(headCNN1D_outChannels3,self.headGRU_hidden_size, self.headGRU_layers, batch_first=True, bidirectional=bool(self.headGRU_directions-1))
        
        # GW_CNN1D Module
        gwCNN1D_outChannels1 = 16
        gwCNN1D_kernelSize1 = 3
        gwCNN1D_featureSize1 = floor((self.gwFeatureLength - gwCNN1D_kernelSize1 + 1)/2)
        gwCNN1D_outChannels2 = 16
        gwCNN1D_kernelSize2 = 3
        gwCNN1D_featureSize2 = floor((gwCNN1D_featureSize1 - gwCNN1D_kernelSize2 + 1)/2)
        gwCNN1D_outChannels3 = 16
        gwCNN1D_kernelSize3 = 3
        gwCNN1D_featureSize3 = floor((gwCNN1D_featureSize2 - gwCNN1D_kernelSize3 + 1)/2)
        self.gwCNN1D_outputSize = gwCNN1D_featureSize3 * gwCNN1D_outChannels3
        self.gwCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gwFeatureNum, out_channels=gwCNN1D_outChannels1,kernel_size=gwCNN1D_kernelSize1),
            nn.BatchNorm1d(gwCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=gwCNN1D_outChannels1, out_channels=gwCNN1D_outChannels2,kernel_size=gwCNN1D_kernelSize2),
            nn.BatchNorm1d(gwCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=gwCNN1D_outChannels2, out_channels=gwCNN1D_outChannels3,kernel_size=gwCNN1D_kernelSize3),
            nn.BatchNorm1d(gwCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),
             )
        
        # GW_GRU Module
        self.gwGRU_hidden_size = 64
        self.gwGRU_layers = 1
        self.gwGRU_directions = 2
        self.gwGRU = nn.GRU(gwCNN1D_outChannels3,self.gwGRU_hidden_size, self.gwGRU_layers, batch_first=True, bidirectional=bool(self.gwGRU_directions-1))
        
        # task prediction FC Module
        headGRU_output_size = self.headGRU_hidden_size*self.headGRU_directions
        gwGRU_output_size = self.gwGRU_hidden_size*self.gwGRU_directions
        prdFC_inputSize = headGRU_output_size + gwGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        headFeature = x[:, index: index+self.headFeatureSize]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
                
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        headFeature = headFeature.permute(0,2,1)        
        headFeatureOut = self.headCNN1D(headFeature)
        headFeatureOut = headFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.headGRU_layers*self.headGRU_directions, x.size(0), self.headGRU_hidden_size).to(device) 
        headGruOut, _ = self.headGRU(headFeatureOut, h0)  
        headGruOut = headGruOut[:, -1, :]
        
        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        gwFeature = gwFeature.permute(0,2,1)        
        gwFeatureOut = self.gwCNN1D(gwFeature)
        gwFeatureOut = gwFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.gwGRU_layers*self.gwGRU_directions, x.size(0), self.gwGRU_hidden_size).to(device) 
        gwGruOut, _ = self.gwGRU(gwFeatureOut, h0)  
        gwGruOut = gwGruOut[:, -1, :]
        
        out = torch.cat((headGruOut, gwGruOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out  

    
# EHTask model using only CNN module.
class EHTask_CNN(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Head feature size: {}'.format(self.headFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_CNN1D Module
        eyeCNN1D_outChannels1 = 16
        eyeCNN1D_kernelSize1 = 3
        eyeCNN1D_featureSize1 = floor((self.eyeFeatureLength - eyeCNN1D_kernelSize1 + 1)/2)
        eyeCNN1D_outChannels2 = 16
        eyeCNN1D_kernelSize2 = 3
        eyeCNN1D_featureSize2 = floor((eyeCNN1D_featureSize1 - eyeCNN1D_kernelSize2 + 1)/2)
        eyeCNN1D_outChannels3 = 16
        eyeCNN1D_kernelSize3 = 3
        eyeCNN1D_featureSize3 = floor((eyeCNN1D_featureSize2 - eyeCNN1D_kernelSize3 + 1)/2)
        self.eyeCNN1D_outputSize = eyeCNN1D_featureSize3 * eyeCNN1D_outChannels3
        self.eyeCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.eyeFeatureNum, out_channels=eyeCNN1D_outChannels1,kernel_size=eyeCNN1D_kernelSize1),
            nn.BatchNorm1d(eyeCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=eyeCNN1D_outChannels1, out_channels=eyeCNN1D_outChannels2,kernel_size=eyeCNN1D_kernelSize2),
            nn.BatchNorm1d(eyeCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=eyeCNN1D_outChannels2, out_channels=eyeCNN1D_outChannels3,kernel_size=eyeCNN1D_kernelSize3),
            nn.BatchNorm1d(eyeCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
                
        # Head_CNN1D Module
        headCNN1D_outChannels1 = 16
        headCNN1D_kernelSize1 = 3
        headCNN1D_featureSize1 = floor((self.headFeatureLength - headCNN1D_kernelSize1 + 1)/2)
        headCNN1D_outChannels2 = 16
        headCNN1D_kernelSize2 = 3
        headCNN1D_featureSize2 = floor((headCNN1D_featureSize1 - headCNN1D_kernelSize2 + 1)/2)
        headCNN1D_outChannels3 = 16
        headCNN1D_kernelSize3 = 3
        headCNN1D_featureSize3 = floor((headCNN1D_featureSize2 - headCNN1D_kernelSize3 + 1)/2)
        self.headCNN1D_outputSize = headCNN1D_featureSize3 * headCNN1D_outChannels3
        self.headCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headCNN1D_outChannels1,kernel_size=headCNN1D_kernelSize1),
            nn.BatchNorm1d(headCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=headCNN1D_outChannels1, out_channels=headCNN1D_outChannels2,kernel_size=headCNN1D_kernelSize2),
            nn.BatchNorm1d(headCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=headCNN1D_outChannels2, out_channels=headCNN1D_outChannels3,kernel_size=headCNN1D_kernelSize3),
            nn.BatchNorm1d(headCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),       
             )
                
        # GW_CNN1D Module
        gwCNN1D_outChannels1 = 16
        gwCNN1D_kernelSize1 = 3
        gwCNN1D_featureSize1 = floor((self.gwFeatureLength - gwCNN1D_kernelSize1 + 1)/2)
        gwCNN1D_outChannels2 = 16
        gwCNN1D_kernelSize2 = 3
        gwCNN1D_featureSize2 = floor((gwCNN1D_featureSize1 - gwCNN1D_kernelSize2 + 1)/2)
        gwCNN1D_outChannels3 = 16
        gwCNN1D_kernelSize3 = 3
        gwCNN1D_featureSize3 = floor((gwCNN1D_featureSize2 - gwCNN1D_kernelSize3 + 1)/2)
        self.gwCNN1D_outputSize = gwCNN1D_featureSize3 * gwCNN1D_outChannels3
        self.gwCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gwFeatureNum, out_channels=gwCNN1D_outChannels1,kernel_size=gwCNN1D_kernelSize1),
            nn.BatchNorm1d(gwCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=gwCNN1D_outChannels1, out_channels=gwCNN1D_outChannels2,kernel_size=gwCNN1D_kernelSize2),
            nn.BatchNorm1d(gwCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=gwCNN1D_outChannels2, out_channels=gwCNN1D_outChannels3,kernel_size=gwCNN1D_kernelSize3),
            nn.BatchNorm1d(gwCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),
             )
                
        # task prediction FC Module
        prdFC_inputSize = self.eyeCNN1D_outputSize + self.headCNN1D_outputSize + self.gwCNN1D_outputSize
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        headFeature = x[:, index: index+self.headFeatureSize]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)
        eyeFeature = eyeFeature.permute(0,2,1)        
        eyeFeatureOut = self.eyeCNN1D(eyeFeature)
        eyeCNNOut = eyeFeatureOut.reshape(-1, self.eyeCNN1D_outputSize)
        
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        headFeature = headFeature.permute(0,2,1)        
        headFeatureOut = self.headCNN1D(headFeature)
        headCNNOut = headFeatureOut.reshape(-1, self.headCNN1D_outputSize)
        
        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        gwFeature = gwFeature.permute(0,2,1)        
        gwFeatureOut = self.gwCNN1D(gwFeature)
        gwCNNOut = gwFeatureOut.reshape(-1, self.gwCNN1D_outputSize)
        
        out = torch.cat((eyeCNNOut, headCNNOut), 1)
        out = torch.cat((out, gwCNNOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out          
    
    
# EHTask model using only BiGRU module.
class EHTask_BiGRU(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Head feature size: {}'.format(self.headFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_GRU Module
        self.eyeGRU_hidden_size = 64
        self.eyeGRU_layers = 1
        self.eyeGRU_directions = 2
        self.eyeGRU = nn.GRU(self.eyeFeatureNum,self.eyeGRU_hidden_size, self.eyeGRU_layers, batch_first=True, bidirectional=bool(self.eyeGRU_directions-1))
        
        # Head_GRU Module
        self.headGRU_hidden_size = 64
        self.headGRU_layers = 1
        self.headGRU_directions = 2
        self.headGRU = nn.GRU(self.headFeatureNum,self.headGRU_hidden_size, self.headGRU_layers, batch_first=True, bidirectional=bool(self.headGRU_directions-1))
                
        # GW_GRU Module
        self.gwGRU_hidden_size = 64
        self.gwGRU_layers = 1
        self.gwGRU_directions = 2
        self.gwGRU = nn.GRU(self.gwFeatureNum,self.gwGRU_hidden_size, self.gwGRU_layers, batch_first=True, bidirectional=bool(self.gwGRU_directions-1))
        
        # task prediction FC Module
        eyeGRU_output_size = self.eyeGRU_hidden_size*self.eyeGRU_directions
        headGRU_output_size = self.headGRU_hidden_size*self.headGRU_directions
        gwGRU_output_size = self.gwGRU_hidden_size*self.gwGRU_directions
        prdFC_inputSize = eyeGRU_output_size + headGRU_output_size + gwGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        headFeature = x[:, index: index+self.headFeatureSize]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)        
        h0 = torch.zeros(self.eyeGRU_layers*self.eyeGRU_directions, x.size(0), self.eyeGRU_hidden_size).to(device) 
        eyeGruOut, _ = self.eyeGRU(eyeFeature, h0)  
        eyeGruOut = eyeGruOut[:, -1, :]
        
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        h0 = torch.zeros(self.headGRU_layers*self.headGRU_directions, x.size(0), self.headGRU_hidden_size).to(device) 
        headGruOut, _ = self.headGRU(headFeature, h0)  
        headGruOut = headGruOut[:, -1, :]

        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        h0 = torch.zeros(self.gwGRU_layers*self.gwGRU_directions, x.size(0), self.gwGRU_hidden_size).to(device) 
        gwGruOut, _ = self.gwGRU(gwFeature, h0)  
        gwGruOut = gwGruOut[:, -1, :]
        
        out = torch.cat((eyeGruOut, headGruOut), 1)
        out = torch.cat((out, gwGruOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out          
    
    
# EHTask model using only LSTM module.
class EHTask_LSTM(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Head feature size: {}'.format(self.headFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_LSTM Module
        self.eyeLSTM_hidden_size = 64
        self.eyeLSTM_layers = 1
        self.eyeLSTM_directions = 1
        self.eyeLSTM = nn.LSTM(self.eyeFeatureNum,self.eyeLSTM_hidden_size, self.eyeLSTM_layers, batch_first=True, bidirectional=bool(self.eyeLSTM_directions-1))
        
        # Head_LSTM Module
        self.headLSTM_hidden_size = 64
        self.headLSTM_layers = 1
        self.headLSTM_directions = 1
        self.headLSTM = nn.LSTM(self.headFeatureNum,self.headLSTM_hidden_size, self.headLSTM_layers, batch_first=True, bidirectional=bool(self.headLSTM_directions-1))
                
        # GW_LSTM Module
        self.gwLSTM_hidden_size = 64
        self.gwLSTM_layers = 1
        self.gwLSTM_directions = 1
        self.gwLSTM = nn.LSTM(self.gwFeatureNum,self.gwLSTM_hidden_size, self.gwLSTM_layers, batch_first=True, bidirectional=bool(self.gwLSTM_directions-1))
        
        # task prediction FC Module
        eyeLSTM_output_size = self.eyeLSTM_hidden_size*self.eyeLSTM_directions
        headLSTM_output_size = self.headLSTM_hidden_size*self.headLSTM_directions
        gwLSTM_output_size = self.gwLSTM_hidden_size*self.gwLSTM_directions
        prdFC_inputSize = eyeLSTM_output_size + headLSTM_output_size + gwLSTM_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        headFeature = x[:, index: index+self.headFeatureSize]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)     
        h0 = torch.zeros(self.eyeLSTM_layers*self.eyeLSTM_directions, x.size(0), self.eyeLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.eyeLSTM_layers*self.eyeLSTM_directions, x.size(0), self.eyeLSTM_hidden_size).to(device)
        eyeLSTMOut, _ = self.eyeLSTM(eyeFeature, (h0, c0))
        eyeLSTMOut = eyeLSTMOut[:, -1, :]
                
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        h0 = torch.zeros(self.headLSTM_layers*self.headLSTM_directions, x.size(0), self.headLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.headLSTM_layers*self.headLSTM_directions, x.size(0), self.headLSTM_hidden_size).to(device)
        headLSTMOut, _ = self.headLSTM(headFeature, (h0, c0))
        headLSTMOut = headLSTMOut[:, -1, :]

        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        h0 = torch.zeros(self.gwLSTM_layers*self.gwLSTM_directions, x.size(0), self.gwLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.gwLSTM_layers*self.gwLSTM_directions, x.size(0), self.gwLSTM_hidden_size).to(device)
        gwLSTMOut, _ = self.gwLSTM(gwFeature, (h0, c0))
        gwLSTMOut = gwLSTMOut[:, -1, :]
        
        out = torch.cat((eyeLSTMOut, headLSTMOut), 1)
        out = torch.cat((out, gwLSTMOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out          
    
    
# EHTask model using only BiLSTM module.
class EHTask_BiLSTM(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Head feature size: {}'.format(self.headFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_LSTM Module
        self.eyeLSTM_hidden_size = 64
        self.eyeLSTM_layers = 1
        self.eyeLSTM_directions = 2
        self.eyeLSTM = nn.LSTM(self.eyeFeatureNum,self.eyeLSTM_hidden_size, self.eyeLSTM_layers, batch_first=True, bidirectional=bool(self.eyeLSTM_directions-1))
        
        # Head_LSTM Module
        self.headLSTM_hidden_size = 64
        self.headLSTM_layers = 1
        self.headLSTM_directions = 2
        self.headLSTM = nn.LSTM(self.headFeatureNum,self.headLSTM_hidden_size, self.headLSTM_layers, batch_first=True, bidirectional=bool(self.headLSTM_directions-1))
                
        # GW_LSTM Module
        self.gwLSTM_hidden_size = 64
        self.gwLSTM_layers = 1
        self.gwLSTM_directions = 2
        self.gwLSTM = nn.LSTM(self.gwFeatureNum,self.gwLSTM_hidden_size, self.gwLSTM_layers, batch_first=True, bidirectional=bool(self.gwLSTM_directions-1))
        
        # task prediction FC Module
        eyeLSTM_output_size = self.eyeLSTM_hidden_size*self.eyeLSTM_directions
        headLSTM_output_size = self.headLSTM_hidden_size*self.headLSTM_directions
        gwLSTM_output_size = self.gwLSTM_hidden_size*self.gwLSTM_directions
        prdFC_inputSize = eyeLSTM_output_size + headLSTM_output_size + gwLSTM_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        headFeature = x[:, index: index+self.headFeatureSize]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)     
        h0 = torch.zeros(self.eyeLSTM_layers*self.eyeLSTM_directions, x.size(0), self.eyeLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.eyeLSTM_layers*self.eyeLSTM_directions, x.size(0), self.eyeLSTM_hidden_size).to(device)
        eyeLSTMOut, _ = self.eyeLSTM(eyeFeature, (h0, c0))
        eyeLSTMOut = eyeLSTMOut[:, -1, :]
                
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        h0 = torch.zeros(self.headLSTM_layers*self.headLSTM_directions, x.size(0), self.headLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.headLSTM_layers*self.headLSTM_directions, x.size(0), self.headLSTM_hidden_size).to(device)
        headLSTMOut, _ = self.headLSTM(headFeature, (h0, c0))
        headLSTMOut = headLSTMOut[:, -1, :]

        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        h0 = torch.zeros(self.gwLSTM_layers*self.gwLSTM_directions, x.size(0), self.gwLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.gwLSTM_layers*self.gwLSTM_directions, x.size(0), self.gwLSTM_hidden_size).to(device)
        gwLSTMOut, _ = self.gwLSTM(gwFeature, (h0, c0))
        gwLSTMOut = gwLSTMOut[:, -1, :]
        
        out = torch.cat((eyeLSTMOut, headLSTMOut), 1)
        out = torch.cat((out, gwLSTMOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out          

    
# EHTask model using GRU to replace BiGRU    
class EHTask_CNN_GRU(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Head feature size: {}'.format(self.headFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_CNN1D Module
        eyeCNN1D_outChannels1 = 16
        eyeCNN1D_kernelSize1 = 3
        eyeCNN1D_featureSize1 = floor((self.eyeFeatureLength - eyeCNN1D_kernelSize1 + 1)/2)
        eyeCNN1D_outChannels2 = 16
        eyeCNN1D_kernelSize2 = 3
        eyeCNN1D_featureSize2 = floor((eyeCNN1D_featureSize1 - eyeCNN1D_kernelSize2 + 1)/2)
        eyeCNN1D_outChannels3 = 16
        eyeCNN1D_kernelSize3 = 3
        eyeCNN1D_featureSize3 = floor((eyeCNN1D_featureSize2 - eyeCNN1D_kernelSize3 + 1)/2)
        self.eyeCNN1D_outputSize = eyeCNN1D_featureSize3 * eyeCNN1D_outChannels3
        self.eyeCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.eyeFeatureNum, out_channels=eyeCNN1D_outChannels1,kernel_size=eyeCNN1D_kernelSize1),
            nn.BatchNorm1d(eyeCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=eyeCNN1D_outChannels1, out_channels=eyeCNN1D_outChannels2,kernel_size=eyeCNN1D_kernelSize2),
            nn.BatchNorm1d(eyeCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=eyeCNN1D_outChannels2, out_channels=eyeCNN1D_outChannels3,kernel_size=eyeCNN1D_kernelSize3),
            nn.BatchNorm1d(eyeCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
        
        # Eye_GRU Module
        self.eyeGRU_hidden_size = 64
        self.eyeGRU_layers = 1
        self.eyeGRU_directions = 1
        self.eyeGRU = nn.GRU(eyeCNN1D_outChannels3,self.eyeGRU_hidden_size, self.eyeGRU_layers, batch_first=True, bidirectional=bool(self.eyeGRU_directions-1))
        
        # Head_CNN1D Module
        headCNN1D_outChannels1 = 16
        headCNN1D_kernelSize1 = 3
        headCNN1D_featureSize1 = floor((self.headFeatureLength - headCNN1D_kernelSize1 + 1)/2)
        headCNN1D_outChannels2 = 16
        headCNN1D_kernelSize2 = 3
        headCNN1D_featureSize2 = floor((headCNN1D_featureSize1 - headCNN1D_kernelSize2 + 1)/2)
        headCNN1D_outChannels3 = 16
        headCNN1D_kernelSize3 = 3
        headCNN1D_featureSize3 = floor((headCNN1D_featureSize2 - headCNN1D_kernelSize3 + 1)/2)
        self.headCNN1D_outputSize = headCNN1D_featureSize3 * headCNN1D_outChannels3
        self.headCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headCNN1D_outChannels1,kernel_size=headCNN1D_kernelSize1),
            nn.BatchNorm1d(headCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=headCNN1D_outChannels1, out_channels=headCNN1D_outChannels2,kernel_size=headCNN1D_kernelSize2),
            nn.BatchNorm1d(headCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=headCNN1D_outChannels2, out_channels=headCNN1D_outChannels3,kernel_size=headCNN1D_kernelSize3),
            nn.BatchNorm1d(headCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),       
             )
        
        # Head_GRU Module
        self.headGRU_hidden_size = 64
        self.headGRU_layers = 1
        self.headGRU_directions = 1
        self.headGRU = nn.GRU(headCNN1D_outChannels3,self.headGRU_hidden_size, self.headGRU_layers, batch_first=True, bidirectional=bool(self.headGRU_directions-1))
        
        # GW_CNN1D Module
        gwCNN1D_outChannels1 = 16
        gwCNN1D_kernelSize1 = 3
        gwCNN1D_featureSize1 = floor((self.gwFeatureLength - gwCNN1D_kernelSize1 + 1)/2)
        gwCNN1D_outChannels2 = 16
        gwCNN1D_kernelSize2 = 3
        gwCNN1D_featureSize2 = floor((gwCNN1D_featureSize1 - gwCNN1D_kernelSize2 + 1)/2)
        gwCNN1D_outChannels3 = 16
        gwCNN1D_kernelSize3 = 3
        gwCNN1D_featureSize3 = floor((gwCNN1D_featureSize2 - gwCNN1D_kernelSize3 + 1)/2)
        self.gwCNN1D_outputSize = gwCNN1D_featureSize3 * gwCNN1D_outChannels3
        self.gwCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gwFeatureNum, out_channels=gwCNN1D_outChannels1,kernel_size=gwCNN1D_kernelSize1),
            nn.BatchNorm1d(gwCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=gwCNN1D_outChannels1, out_channels=gwCNN1D_outChannels2,kernel_size=gwCNN1D_kernelSize2),
            nn.BatchNorm1d(gwCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=gwCNN1D_outChannels2, out_channels=gwCNN1D_outChannels3,kernel_size=gwCNN1D_kernelSize3),
            nn.BatchNorm1d(gwCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),
             )
        
        # GW_GRU Module
        self.gwGRU_hidden_size = 64
        self.gwGRU_layers = 1
        self.gwGRU_directions = 1
        self.gwGRU = nn.GRU(gwCNN1D_outChannels3,self.gwGRU_hidden_size, self.gwGRU_layers, batch_first=True, bidirectional=bool(self.gwGRU_directions-1))
        
        # task prediction FC Module
        eyeGRU_output_size = self.eyeGRU_hidden_size*self.eyeGRU_directions
        headGRU_output_size = self.headGRU_hidden_size*self.headGRU_directions
        gwGRU_output_size = self.gwGRU_hidden_size*self.gwGRU_directions
        prdFC_inputSize = eyeGRU_output_size + headGRU_output_size + gwGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        headFeature = x[:, index: index+self.headFeatureSize]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)
        eyeFeature = eyeFeature.permute(0,2,1)        
        eyeFeatureOut = self.eyeCNN1D(eyeFeature)
        eyeFeatureOut = eyeFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.eyeGRU_layers*self.eyeGRU_directions, x.size(0), self.eyeGRU_hidden_size).to(device) 
        eyeGruOut, _ = self.eyeGRU(eyeFeatureOut, h0)  
        eyeGruOut = eyeGruOut[:, -1, :]
        
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        headFeature = headFeature.permute(0,2,1)        
        headFeatureOut = self.headCNN1D(headFeature)
        headFeatureOut = headFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.headGRU_layers*self.headGRU_directions, x.size(0), self.headGRU_hidden_size).to(device) 
        headGruOut, _ = self.headGRU(headFeatureOut, h0)  
        headGruOut = headGruOut[:, -1, :]

        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        gwFeature = gwFeature.permute(0,2,1)        
        gwFeatureOut = self.gwCNN1D(gwFeature)
        gwFeatureOut = gwFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.gwGRU_layers*self.gwGRU_directions, x.size(0), self.gwGRU_hidden_size).to(device) 
        gwGruOut, _ = self.gwGRU(gwFeatureOut, h0)  
        gwGruOut = gwGruOut[:, -1, :]
        
        out = torch.cat((eyeGruOut, headGruOut), 1)
        out = torch.cat((out, gwGruOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out   
    
    
# EHTask model using LSTM to replace BiGRU
class EHTask_CNN_LSTM(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Head feature size: {}'.format(self.headFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_CNN1D Module
        eyeCNN1D_outChannels1 = 16
        eyeCNN1D_kernelSize1 = 3
        eyeCNN1D_featureSize1 = floor((self.eyeFeatureLength - eyeCNN1D_kernelSize1 + 1)/2)
        eyeCNN1D_outChannels2 = 16
        eyeCNN1D_kernelSize2 = 3
        eyeCNN1D_featureSize2 = floor((eyeCNN1D_featureSize1 - eyeCNN1D_kernelSize2 + 1)/2)
        eyeCNN1D_outChannels3 = 16
        eyeCNN1D_kernelSize3 = 3
        eyeCNN1D_featureSize3 = floor((eyeCNN1D_featureSize2 - eyeCNN1D_kernelSize3 + 1)/2)
        self.eyeCNN1D_outputSize = eyeCNN1D_featureSize3 * eyeCNN1D_outChannels3
        self.eyeCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.eyeFeatureNum, out_channels=eyeCNN1D_outChannels1,kernel_size=eyeCNN1D_kernelSize1),
            nn.BatchNorm1d(eyeCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=eyeCNN1D_outChannels1, out_channels=eyeCNN1D_outChannels2,kernel_size=eyeCNN1D_kernelSize2),
            nn.BatchNorm1d(eyeCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=eyeCNN1D_outChannels2, out_channels=eyeCNN1D_outChannels3,kernel_size=eyeCNN1D_kernelSize3),
            nn.BatchNorm1d(eyeCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
        
        # Eye_LSTM Module
        self.eyeLSTM_hidden_size = 64
        self.eyeLSTM_layers = 1
        self.eyeLSTM_directions = 1
        self.eyeLSTM = nn.LSTM(eyeCNN1D_outChannels3,self.eyeLSTM_hidden_size, self.eyeLSTM_layers, batch_first=True, bidirectional=bool(self.eyeLSTM_directions-1))
        
        # Head_CNN1D Module
        headCNN1D_outChannels1 = 16
        headCNN1D_kernelSize1 = 3
        headCNN1D_featureSize1 = floor((self.headFeatureLength - headCNN1D_kernelSize1 + 1)/2)
        headCNN1D_outChannels2 = 16
        headCNN1D_kernelSize2 = 3
        headCNN1D_featureSize2 = floor((headCNN1D_featureSize1 - headCNN1D_kernelSize2 + 1)/2)
        headCNN1D_outChannels3 = 16
        headCNN1D_kernelSize3 = 3
        headCNN1D_featureSize3 = floor((headCNN1D_featureSize2 - headCNN1D_kernelSize3 + 1)/2)
        self.headCNN1D_outputSize = headCNN1D_featureSize3 * headCNN1D_outChannels3
        self.headCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headCNN1D_outChannels1,kernel_size=headCNN1D_kernelSize1),
            nn.BatchNorm1d(headCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=headCNN1D_outChannels1, out_channels=headCNN1D_outChannels2,kernel_size=headCNN1D_kernelSize2),
            nn.BatchNorm1d(headCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=headCNN1D_outChannels2, out_channels=headCNN1D_outChannels3,kernel_size=headCNN1D_kernelSize3),
            nn.BatchNorm1d(headCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),       
             )
        
        # Head_LSTM Module
        self.headLSTM_hidden_size = 64
        self.headLSTM_layers = 1
        self.headLSTM_directions = 1
        self.headLSTM = nn.LSTM(headCNN1D_outChannels3,self.headLSTM_hidden_size, self.headLSTM_layers, batch_first=True, bidirectional=bool(self.headLSTM_directions-1))
        
        # GW_CNN1D Module
        gwCNN1D_outChannels1 = 16
        gwCNN1D_kernelSize1 = 3
        gwCNN1D_featureSize1 = floor((self.gwFeatureLength - gwCNN1D_kernelSize1 + 1)/2)
        gwCNN1D_outChannels2 = 16
        gwCNN1D_kernelSize2 = 3
        gwCNN1D_featureSize2 = floor((gwCNN1D_featureSize1 - gwCNN1D_kernelSize2 + 1)/2)
        gwCNN1D_outChannels3 = 16
        gwCNN1D_kernelSize3 = 3
        gwCNN1D_featureSize3 = floor((gwCNN1D_featureSize2 - gwCNN1D_kernelSize3 + 1)/2)
        self.gwCNN1D_outputSize = gwCNN1D_featureSize3 * gwCNN1D_outChannels3
        self.gwCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gwFeatureNum, out_channels=gwCNN1D_outChannels1,kernel_size=gwCNN1D_kernelSize1),
            nn.BatchNorm1d(gwCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=gwCNN1D_outChannels1, out_channels=gwCNN1D_outChannels2,kernel_size=gwCNN1D_kernelSize2),
            nn.BatchNorm1d(gwCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=gwCNN1D_outChannels2, out_channels=gwCNN1D_outChannels3,kernel_size=gwCNN1D_kernelSize3),
            nn.BatchNorm1d(gwCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),
             )
        
        # GW_LSTM Module
        self.gwLSTM_hidden_size = 64
        self.gwLSTM_layers = 1
        self.gwLSTM_directions = 1
        self.gwLSTM = nn.LSTM(gwCNN1D_outChannels3,self.gwLSTM_hidden_size, self.gwLSTM_layers, batch_first=True, bidirectional=bool(self.gwLSTM_directions-1))
        
        # task prediction FC Module
        eyeLSTM_output_size = self.eyeLSTM_hidden_size*self.eyeLSTM_directions
        headLSTM_output_size = self.headLSTM_hidden_size*self.headLSTM_directions
        gwLSTM_output_size = self.gwLSTM_hidden_size*self.gwLSTM_directions
        prdFC_inputSize = eyeLSTM_output_size + headLSTM_output_size + gwLSTM_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        headFeature = x[:, index: index+self.headFeatureSize]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)
        eyeFeature = eyeFeature.permute(0,2,1)        
        eyeFeatureOut = self.eyeCNN1D(eyeFeature)
        eyeFeatureOut = eyeFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.eyeLSTM_layers*self.eyeLSTM_directions, x.size(0), self.eyeLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.eyeLSTM_layers*self.eyeLSTM_directions, x.size(0), self.eyeLSTM_hidden_size).to(device)
        eyeLSTMOut, _ = self.eyeLSTM(eyeFeatureOut, (h0, c0))
        eyeLSTMOut = eyeLSTMOut[:, -1, :]
        
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        headFeature = headFeature.permute(0,2,1)        
        headFeatureOut = self.headCNN1D(headFeature)
        headFeatureOut = headFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.headLSTM_layers*self.headLSTM_directions, x.size(0), self.headLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.headLSTM_layers*self.headLSTM_directions, x.size(0), self.headLSTM_hidden_size).to(device)
        headLSTMOut, _ = self.headLSTM(headFeatureOut, (h0, c0))  
        headLSTMOut = headLSTMOut[:, -1, :]

        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        gwFeature = gwFeature.permute(0,2,1)        
        gwFeatureOut = self.gwCNN1D(gwFeature)
        gwFeatureOut = gwFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.gwLSTM_layers*self.gwLSTM_directions, x.size(0), self.gwLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.gwLSTM_layers*self.gwLSTM_directions, x.size(0), self.gwLSTM_hidden_size).to(device)
        gwLSTMOut, _ = self.gwLSTM(gwFeatureOut, (h0, c0))  
        gwLSTMOut = gwLSTMOut[:, -1, :]
        
        out = torch.cat((eyeLSTMOut, headLSTMOut), 1)
        out = torch.cat((out, gwLSTMOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out        
    

# EHTask model using BiLSTM to replace BiGRU
class EHTask_CNN_BiLSTM(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        print('Eye-in-head feature size: {}'.format(self.eyeFeatureSize))
        print('Head feature size: {}'.format(self.headFeatureSize))
        print('Gaze-in-world feature size: {}'.format(self.gwFeatureSize))
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_CNN1D Module
        eyeCNN1D_outChannels1 = 16
        eyeCNN1D_kernelSize1 = 3
        eyeCNN1D_featureSize1 = floor((self.eyeFeatureLength - eyeCNN1D_kernelSize1 + 1)/2)
        eyeCNN1D_outChannels2 = 16
        eyeCNN1D_kernelSize2 = 3
        eyeCNN1D_featureSize2 = floor((eyeCNN1D_featureSize1 - eyeCNN1D_kernelSize2 + 1)/2)
        eyeCNN1D_outChannels3 = 16
        eyeCNN1D_kernelSize3 = 3
        eyeCNN1D_featureSize3 = floor((eyeCNN1D_featureSize2 - eyeCNN1D_kernelSize3 + 1)/2)
        self.eyeCNN1D_outputSize = eyeCNN1D_featureSize3 * eyeCNN1D_outChannels3
        self.eyeCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.eyeFeatureNum, out_channels=eyeCNN1D_outChannels1,kernel_size=eyeCNN1D_kernelSize1),
            nn.BatchNorm1d(eyeCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=eyeCNN1D_outChannels1, out_channels=eyeCNN1D_outChannels2,kernel_size=eyeCNN1D_kernelSize2),
            nn.BatchNorm1d(eyeCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=eyeCNN1D_outChannels2, out_channels=eyeCNN1D_outChannels3,kernel_size=eyeCNN1D_kernelSize3),
            nn.BatchNorm1d(eyeCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
        
        # Eye_LSTM Module
        self.eyeLSTM_hidden_size = 64
        self.eyeLSTM_layers = 1
        self.eyeLSTM_directions = 2
        self.eyeLSTM = nn.LSTM(eyeCNN1D_outChannels3,self.eyeLSTM_hidden_size, self.eyeLSTM_layers, batch_first=True, bidirectional=bool(self.eyeLSTM_directions-1))
        
        # Head_CNN1D Module
        headCNN1D_outChannels1 = 16
        headCNN1D_kernelSize1 = 3
        headCNN1D_featureSize1 = floor((self.headFeatureLength - headCNN1D_kernelSize1 + 1)/2)
        headCNN1D_outChannels2 = 16
        headCNN1D_kernelSize2 = 3
        headCNN1D_featureSize2 = floor((headCNN1D_featureSize1 - headCNN1D_kernelSize2 + 1)/2)
        headCNN1D_outChannels3 = 16
        headCNN1D_kernelSize3 = 3
        headCNN1D_featureSize3 = floor((headCNN1D_featureSize2 - headCNN1D_kernelSize3 + 1)/2)
        self.headCNN1D_outputSize = headCNN1D_featureSize3 * headCNN1D_outChannels3
        self.headCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headCNN1D_outChannels1,kernel_size=headCNN1D_kernelSize1),
            nn.BatchNorm1d(headCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=headCNN1D_outChannels1, out_channels=headCNN1D_outChannels2,kernel_size=headCNN1D_kernelSize2),
            nn.BatchNorm1d(headCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=headCNN1D_outChannels2, out_channels=headCNN1D_outChannels3,kernel_size=headCNN1D_kernelSize3),
            nn.BatchNorm1d(headCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),       
             )
        
        # Head_LSTM Module
        self.headLSTM_hidden_size = 64
        self.headLSTM_layers = 1
        self.headLSTM_directions = 2
        self.headLSTM = nn.LSTM(headCNN1D_outChannels3,self.headLSTM_hidden_size, self.headLSTM_layers, batch_first=True, bidirectional=bool(self.headLSTM_directions-1))
        
        # GW_CNN1D Module
        gwCNN1D_outChannels1 = 16
        gwCNN1D_kernelSize1 = 3
        gwCNN1D_featureSize1 = floor((self.gwFeatureLength - gwCNN1D_kernelSize1 + 1)/2)
        gwCNN1D_outChannels2 = 16
        gwCNN1D_kernelSize2 = 3
        gwCNN1D_featureSize2 = floor((gwCNN1D_featureSize1 - gwCNN1D_kernelSize2 + 1)/2)
        gwCNN1D_outChannels3 = 16
        gwCNN1D_kernelSize3 = 3
        gwCNN1D_featureSize3 = floor((gwCNN1D_featureSize2 - gwCNN1D_kernelSize3 + 1)/2)
        self.gwCNN1D_outputSize = gwCNN1D_featureSize3 * gwCNN1D_outChannels3
        self.gwCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gwFeatureNum, out_channels=gwCNN1D_outChannels1,kernel_size=gwCNN1D_kernelSize1),
            nn.BatchNorm1d(gwCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=gwCNN1D_outChannels1, out_channels=gwCNN1D_outChannels2,kernel_size=gwCNN1D_kernelSize2),
            nn.BatchNorm1d(gwCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=gwCNN1D_outChannels2, out_channels=gwCNN1D_outChannels3,kernel_size=gwCNN1D_kernelSize3),
            nn.BatchNorm1d(gwCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),
             )
        
        # GW_LSTM Module
        self.gwLSTM_hidden_size = 64
        self.gwLSTM_layers = 1
        self.gwLSTM_directions = 2
        self.gwLSTM = nn.LSTM(gwCNN1D_outChannels3,self.gwLSTM_hidden_size, self.gwLSTM_layers, batch_first=True, bidirectional=bool(self.gwLSTM_directions-1))
        
        # task prediction FC Module
        eyeLSTM_output_size = self.eyeLSTM_hidden_size*self.eyeLSTM_directions
        headLSTM_output_size = self.headLSTM_hidden_size*self.headLSTM_directions
        gwLSTM_output_size = self.gwLSTM_hidden_size*self.gwLSTM_directions
        prdFC_inputSize = eyeLSTM_output_size + headLSTM_output_size + gwLSTM_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        index = self.eyeFeatureSize
        eyeFeature = x[:, 0:index]
        headFeature = x[:, index: index+self.headFeatureSize]
        index = index+self.headFeatureSize
        gwFeature = x[:, index: index+self.gwFeatureSize]
        
        eyeFeature = eyeFeature.reshape(-1, self.eyeFeatureLength, self.eyeFeatureNum)
        eyeFeature = eyeFeature.permute(0,2,1)        
        eyeFeatureOut = self.eyeCNN1D(eyeFeature)
        eyeFeatureOut = eyeFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.eyeLSTM_layers*self.eyeLSTM_directions, x.size(0), self.eyeLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.eyeLSTM_layers*self.eyeLSTM_directions, x.size(0), self.eyeLSTM_hidden_size).to(device)
        eyeLSTMOut, _ = self.eyeLSTM(eyeFeatureOut, (h0, c0))
        eyeLSTMOut = eyeLSTMOut[:, -1, :]
        
        headFeature = headFeature.reshape(-1, self.headFeatureLength, self.headFeatureNum)
        headFeature = headFeature.permute(0,2,1)        
        headFeatureOut = self.headCNN1D(headFeature)
        headFeatureOut = headFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.headLSTM_layers*self.headLSTM_directions, x.size(0), self.headLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.headLSTM_layers*self.headLSTM_directions, x.size(0), self.headLSTM_hidden_size).to(device)
        headLSTMOut, _ = self.headLSTM(headFeatureOut, (h0, c0))  
        headLSTMOut = headLSTMOut[:, -1, :]

        gwFeature = gwFeature.reshape(-1, self.gwFeatureLength, self.gwFeatureNum)
        gwFeature = gwFeature.permute(0,2,1)        
        gwFeatureOut = self.gwCNN1D(gwFeature)
        gwFeatureOut = gwFeatureOut.permute(0,2,1)        
        h0 = torch.zeros(self.gwLSTM_layers*self.gwLSTM_directions, x.size(0), self.gwLSTM_hidden_size).to(device)
        c0 = torch.zeros(self.gwLSTM_layers*self.gwLSTM_directions, x.size(0), self.gwLSTM_hidden_size).to(device)
        gwLSTMOut, _ = self.gwLSTM(gwFeatureOut, (h0, c0))  
        gwLSTMOut = gwLSTMOut[:, -1, :]
        
        out = torch.cat((eyeLSTMOut, headLSTMOut), 1)
        out = torch.cat((out, gwLSTMOut), 1)
        out = self.PrdFC(out)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out            