## Solution Explanation

'EHTask_Unity_Example' contains an example of running EHTask model with Unity.

"EHTask_Server.py" loads a pre-trained EHTask model and then waits for input data from Unity client to run the EHTask model.
Note that we only utilize eye-in-head gaze data and head velocity data as input features because gaze-in-world gaze data is only meaningful for 360-degree videos.

"Unity_Client.unity" collects eye-in-head gaze data and head velocity data and sends the data to the python server, i.e. "EHTask_Server.py".

"Unity_Client/Assets/Plugins/" contains the required netmq plugins.

Unity Scripts:  
"CalculateHeadVelocity.cs": calculates the velocity of a head camera.  
"DataRecorder.cs": collects the eye and head data.  
"Client.cs": sends the collected data to a python server.  


Using this example, you can do some interesting things:
1. Collect gaze and head data from Unity and send them to a python server.  
2. Retrain our model and apply it to your Unity scene.  
3. Collect your own data to train your own model.  


## Requirements:
Unity 2019.4.13+  
python 3.6+  
pytorch 1.1.0+  
pyzmq  
netmq  


## Usage:
Step 1: Run "EHTask_Server/EHTask_Server.py".

Step 2: Use Unity to open "Unity_Client" and run "Unity_Client/Assets/Unity_Client.unity".

