# Behavioural Cloning Project

---
## Project Overview

*The objective of the project is to train a model to drive a car autonomously on a simulated track. 
The ability of the model to drive the car is learned from cloning the behaviour of a human driver.
Training data is collected from recordings human driving in the simulator, then fed into a deep learning network which learns the response (steering angle) for every encountered frame in the simulation. The model is then validated on a new track to check for generalization of the learned features for performing steering angle prediction.*

This project is influenced by [nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), [comma.ai paper](https://arxiv.org/pdf/1608.01230v1.pdf), [vivek's blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.5dpi87xzi) and [various posts](https://medium.com/self-driving-cars/6-different-end-to-end-neural-networks-f307fa2904a5#.yk2a9g6ui) from other Udacity students which I consulted while working on my solution. The [Keras Deep Learning library](https://keras.io/) was used with [Tensorflow](https://www.tensorflow.org/) backend to perform deep learning operations. The training was performed on [Amazon EC2 GPU instances](https://aws.amazon.com/ec2/Elastic-GPUs/).

---
## Objectives
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Script excecution instructions
The following resources can be found in this github repository:
* model.py: used to create and train the model
* model.h5: containing a trained convolution neural network 
* preprocess.py: used to preprocess recorded raw images and save the images as HDF5 data files. The preprocessing includes image brightness changes,  normalization, and optioal data augmentation for random shadow generation.
* drive.py: used for driving the car in autonomous mode
* analyze_data.py: used to analyze the distribution of recorded training data, say histogram of steering angle against speed or throttle.
* README.md: project writeup

To run the Using the Udacity provided simulator (earlier one, track 2 was curvy dark road in black mountains) and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python3 drive.py model.h5 [recording_dir]
```

If [recording_dir] is specified, frame images will be automatically saved for later analysis.

