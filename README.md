# EHTask: Recognizing User Tasks from Eye and Head Movements in Immersive Virtual Reality
Project homepage: https://cranehzm.github.io/EHTask.


'EHTask' contains the source code of our model and some pre-trained models.


'EHTask_Unity_Example' contains an example of running EHTask model in Unity.


## Abstract
```
Understanding human visual attention in immersive virtual reality (VR) is crucial for many important applications, including gaze prediction, gaze guidance, and gaze-contingent rendering.
However, previous works on visual attention analysis typically only explored one specific VR task and paid less attention to the differences between different tasks.
Moreover, existing task recognition methods typically focused on 2D viewing conditions and only explored the effectiveness of human eye movements.
We first collect eye and head movements of 30 participants performing four tasks, i.e. Free viewing, Visual search, Saliency, and Track, in 15 360-degree VR videos.
Using this dataset, we analyze the patterns of human eye and head movements and reveal significant differences across different tasks in terms of fixation duration, saccade amplitude, head rotation velocity, and eye-head coordination.
We then propose EHTask -- a novel learning-based method that employs eye and head movements to recognize user tasks in VR.
We show that our method significantly outperforms the state-of-the-art methods derived from 2D viewing conditions both on our dataset (accuracy of 84.4% vs. 62.8%) and on a real-world dataset (61.9% vs. 44.1%).
As such, our work provides meaningful insights into human visual attention under different VR tasks and guides future work on recognizing user tasks in VR.
```	


## Environments:
Ubuntu 18.04  
python 3.6+  
pytorch 1.1.0+  
cudatoolkit 10.0  
cudnn 7.6.5  


## Usage:
Step 1: Download the dataset from our project homepage: https://cranehzm.github.io/EHTask.

Step 2: Run the script "run_EHTask.sh" in "EHTask/scripts" directory to retrain or test our model on our dataset.
		Run the script "run_EHTask_GWDataset.sh" in "EHTask/scripts" directory to retrain or test our model on GW dataset.

## Citation
```bibtex
@article{hu22ehtask,
	author={Hu, Zhiming and Bulling, Andreas and Li, Sheng and Wang, Guoping},
	journal={IEEE Transactions on Visualization and Computer Graphics}, 
	title={EHTask: Recognizing User Tasks From Eye and Head Movements in Immersive Virtual Reality}, 
	year={2023},
	volume={29},
	number={4},
	pages={1992--2004},
	doi={10.1109/TVCG.2021.3138902}}  
```
  

