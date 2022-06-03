# Copyright (c) Hu Zhiming 2022/06/03 jimmyhu@pku.edu.cn All Rights Reserved.

# run a pre-trained EHTask model for a single input data.


from models.EHTaskModels import *
import torch
import numpy as np
import zmq


# model parameters
eyeFeatureSize = 500
headFeatureSize = 500
inputSize = eyeFeatureSize + headFeatureSize
numClasses = 4
# path to a pre-trained model
modelPath = './checkpoint/EHTask_EyeHead/checkpoint_epoch_030.tar'


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


def main():
	# Create the model
	print('\n==> Creating the model...')
	# We only utilize eye-in-head gaze data and head velocity data as input features because gaze-in-world gaze data is only meaningful for 360-degree videos.
	model = EHTask_EyeHead(eyeFeatureSize, headFeatureSize, numClasses)
	model = torch.nn.DataParallel(model)
	if device == torch.device('cuda'):
		checkpoint = torch.load(modelPath)
		print('\nDevice: GPU')
	else:
		checkpoint = torch.load(modelPath, map_location=lambda storage, loc: storage)
		print('\nDevice: CPU')
	model.load_state_dict(checkpoint['model_state_dict'])                                          
	# evaluate mode
	model.eval()
	
	while True:
		#  Wait for next request from client
		message = socket.recv()		
		data = message.decode('utf-8').split(',')
		timeStamp = data[0]
		print("Time Stamp: {}".format(timeStamp))
		features = np.zeros((1, inputSize),  dtype=np.float32)
		for i in range(inputSize):			
			features[0, i] = float(data[i+1])
				
		singleInput = torch.tensor(features, dtype=torch.float32, device=device)			
		# Forward pass
		outputs = model(singleInput)
		_, prediction = torch.max(outputs.data, 1)    
		prediction_npy = prediction.data.cpu().detach().numpy()  
		print('Recognized Task : {}'.format(prediction_npy))
		# Task 0-3: Free viewing, Visual search, Saliency, and Track
		task = str(prediction_npy).encode('utf-8')
		socket.send(task)
		
if __name__ == '__main__':
    main()