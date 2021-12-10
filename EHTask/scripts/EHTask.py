# Copyright (c) Hu Zhiming 2021/04/22 jimmyhu@pku.edu.cn All Rights Reserved.

import sys
sys.path.append('../')
from utils import LoadTrainingData, LoadTestData, RemakeDir, MakeDir, SeedTorch
from models import weight_init
from models.EHTaskModels import *
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time
import datetime
import argparse
import os


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set the random seed to ensure reproducibility
SeedTorch(seed=0)


def main(args):
    # Create the model
    print('\n==> Creating the model...')
    model = EHTask(args.eyeFeatureSize, args.headFeatureSize, args.gwFeatureSize, args.numClasses)
    model.apply(weight_init)
    #print('# Number of Model Parameters:', sum(param.numel() for param in model.parameters()))
    
    # print the parameters
    #for name, parameters in model.named_parameters():
        #print(name, parameters)
        
    model = torch.nn.DataParallel(model)
    if args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
        print('\n==> Loss Function: CrossEntropy')
    
    # train the model
    if args.trainFlag == 1:
        # load the training data
        train_loader = LoadTrainingData(args.datasetDir, args.batchSize)
        # optimizer and loss
        lr = args.learningRate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weightDecay)
        expLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1)
        # training start epoch
        startEpoch = 0
        # remake checkpoint directory
        RemakeDir(args.checkpoint)
            
        # training
        localtime = time.asctime(time.localtime(time.time()))
        print('\nTraining starts at ' + localtime)
        # the number of training steps in an epoch
        stepNum = len(train_loader)
        numEpochs = args.epochs
        startTime = datetime.datetime.now()
        for epoch in range(startEpoch, numEpochs):
            # adjust learning rate
            lr = expLR.optimizer.param_groups[0]["lr"]
            
            print('\nEpoch: {} | LR: {:.16f}'.format(epoch + 1, lr))
            for i, (features, labels) in enumerate(train_loader):  
                # Move tensors to the configured device
                features = features.reshape(-1, args.inputSize).to(device)
                labels = labels.reshape(-1,).to(device)
                #print(features.shape)
                #print(labels.shape)
                
                # Forward pass
                outputs = model(features)
                #print(outputs.shape)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(), 30)
                optimizer.step()
                
                # output the loss
                if (i+1) % int(stepNum/args.lossFrequency) == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, numEpochs, i+1, stepNum, loss.item()))
            
            # adjust learning rate
            expLR.step()
            endTime = datetime.datetime.now()
            totalTrainingTime = (endTime - startTime).seconds/60
            print('\nEpoch [{}/{}], Total Training Time: {:.2f} min'.format(epoch+1, numEpochs, totalTrainingTime))    
            
            # save the checkpoint
            if (epoch +1) % args.interval == 0:
                savePath = os.path.join(args.checkpoint, "checkpoint_epoch_{}.tar".format(str(epoch+1).zfill(3)))
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'lr': lr,
                 }, savePath)

        
        localtime = time.asctime(time.localtime(time.time()))
        print('\nTraining ends at ' + localtime)
                
    # test all the existing models
    # load the existing models to test
    if os.path.isdir(args.checkpoint):
        filelist = os.listdir(args.checkpoint)
        checkpoints = []
        checkpointNum = 0
        for name in filelist:
            # checkpoints are stored as tar files
            if os.path.splitext(name)[-1][1:] == 'tar':
                checkpoints.append(name)
                checkpointNum +=1
        # test the checkpoints
        if checkpointNum:
            print('\nCheckpoint Number : {}'.format(checkpointNum))
            checkpoints.sort()
            # load the test data
            test_loader = LoadTestData(args.datasetDir, args.batchSize)
            # load the test labels
            testY = np.load(args.datasetDir + 'testY.npy')
            testSize = testY.shape[0]
            # save the predictions
            if args.savePrd:
                prdDir = args.prdDir
                RemakeDir(prdDir)
            localtime = time.asctime(time.localtime(time.time()))
            print('\nTest starts at ' + localtime)
            for name in checkpoints:
                print("\n==> Test checkpoint : {}".format(name))
                if device == torch.device('cuda'):
                    checkpoint = torch.load(args.checkpoint + name)
                    print('\nDevice: GPU')
                else:
                    checkpoint = torch.load(args.checkpoint + name, map_location=lambda storage, loc: storage)
                    print('\nDevice: CPU')
                            
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch']
                # the model's predictions
                prdY = []
                # evaluate mode
                model.eval()
                startTime = datetime.datetime.now()
                for i, (features, labels) in enumerate(test_loader): 
                    # Move tensors to the configured device
                    features = features.reshape(-1, args.inputSize).to(device)
                    #labels = labels.reshape(-1, 1).to(device)
                     
                    # Forward pass
                    outputs = model(features)
                    _, predictions = torch.max(outputs.data, 1)
                    
                    # save the predictions
                    predictions_npy = predictions.data.cpu().detach().numpy()  
                    if(len(prdY) >0):
                        prdY = np.concatenate((prdY, predictions_npy))
                    else:
                        prdY = predictions_npy
                        
                endTime = datetime.datetime.now()
                # average predicting time for a single sample.
                avgTime = (endTime - startTime).seconds * 1000/testSize
                print('\nAverage prediction time: {:.8f} ms'.format(avgTime))
                
                # Calculate the prediction accuracy
                chanceAccuracy = 1/args.numClasses*100
                print('Chance Level Accuracy: {:.1f}%'.format(chanceAccuracy))
                prdY = prdY.reshape(-1, 1)
                correct = (testY == prdY).sum()
                #print(testY.shape)
                #print(prdY.shape)
                accuracy = correct/testSize*100
                print('Epoch: {}, Single Window Prediction Accuracy: {:.1f}%'.format(epoch, accuracy))
                
                # Majority voting over the whole recording
                testRecordingLabel = np.load(args.datasetDir + 'testRecordingLabel.npy')
                itemLabel = np.unique(testRecordingLabel)
                #print(itemLabel.shape)
                itemNum = itemLabel.shape[0]
                testY_MV = np.zeros(itemNum)
                prdY_MV = np.zeros(itemNum)
                #print(itemNum)
                for i in range(itemNum):
                    #print(itemLabel[i])
                    index = np.where(testRecordingLabel == itemLabel[i])
                    testY_MV[i] = np.argmax(np.bincount(testY[index]))
                    prdY_MV[i] = np.argmax(np.bincount(prdY[index]))
                #print(testY_MV)
                #print(prdY_MV)
                correct = (testY_MV == prdY_MV).sum()
                accuracy = correct/itemNum*100
                print('Epoch: {}, Majority Voting Prediction Accuracy: {:.1f}%'.format(epoch, accuracy))
                
                # save the prediction results
                if args.savePrd:
                    prdDir = args.prdDir + 'predictions_epoch_{}/'.format(str(epoch).zfill(3))
                    MakeDir(prdDir)
                    predictionResults = np.zeros(shape = (testSize, 3))
                    predictionResults[:, 0] = testY.reshape(-1,)
                    predictionResults[:, 1] = prdY.reshape(-1,)
                    predictionResults[:, 2] = testRecordingLabel.reshape(-1,)
                    np.savetxt(prdDir + 'predictions.txt', predictionResults, fmt="%d")
                  
            localtime = time.asctime(time.localtime(time.time()))
            print('\nTest ends at ' + localtime)   
        else:
            print('\n==> No valid checkpoints in directory {}'.format(args.checkpoint))
    else:
        print('\n==> Invalid checkpoint directory: {}'.format(args.checkpoint))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'EHTask Model')
    # the number of input features
    parser.add_argument('--inputSize', default=1500, type=int,
                        help='the size of input features (default: 1500)')
    # the size of eye-in-head features
    parser.add_argument('--eyeFeatureSize', default=500, type=int,
                        help='the size of eye-in-head features (default: 500)')    
    # the size of head features
    parser.add_argument('--headFeatureSize', default=500, type=int,
                        help='the size of head features (default: 500)')    
    # the size of gaze-in-world features
    parser.add_argument('--gwFeatureSize', default=500, type=int,
                        help='the size of gaze-in-world features (default: 500)')    
    # the number of classes to predict
    parser.add_argument('--numClasses', default=4, type=int,
                        help='the number of classes to predict (default: 4)')
    # the directory that saves the dataset
    parser.add_argument('-d', '--datasetDir', default = '../../TaskDataset/EHTask_Cross_User_5_Fold/Test_Fold_1/', type = str, 
                        help = 'the directory that saves the dataset')
    # trainFlag = 1 means train new models; trainFlag = 0 means test existing models
    parser.add_argument('-t', '--trainFlag', default = 1, type = int, help = 'set the flag to train the model (default: 1)')
    # path to save checkpoint
    parser.add_argument('-c', '--checkpoint', default = '../checkpoint/EHTask_Cross_User_5_Fold/Test_Fold_1/', type = str, 
                        help = 'path to save checkpoint')
    # save the prediction results or not
    parser.add_argument('--savePrd', default = 1, type = int, help = 'save the prediction results (1) or not (0) (default: 0)')
    # the directory that saves the prediction results
    parser.add_argument('-p', '--prdDir', default = '../predictions/EHTask_Cross_User_5_Fold/Test_Fold_1/', type = str, 
                        help = 'the directory that saves the prediction results')
    # the number of total epochs to run
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs to run (default: 30)')
    # the batch size
    parser.add_argument('-b', '--batchSize', default=256, type=int,
                        help='the batch size (default: 256)')
    # the interval that we save the checkpoint
    parser.add_argument('-i', '--interval', default=30, type=int,
                        help='the interval that we save the checkpoint (default: 30 epochs)')
    # the initial learning rate.
    parser.add_argument('--learningRate', default=1e-2, type=float,
                        help='initial learning rate (default: 1e-2)')
    parser.add_argument('--weightDecay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.75,
                        help='Used to decay learning rate (default: 0.75)')
    # the loss function.
    parser.add_argument('--loss', default="CrossEntropy", type=str,
                        help='Loss function to train the network (default: CrossEntropy)')
    # the frequency that we output the loss in an epoch.
    parser.add_argument('--lossFrequency', default=3, type=int,
                        help='the frequency that we output the loss in an epoch (default: 3)')
    main(parser.parse_args())
    