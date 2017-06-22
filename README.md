# Behavioural Cloning Project

---
## Project Overview

*The objective of the project is to train a model to drive a car autonomously on a simulated track. 
The ability of the model to drive the car is learned from cloning the behaviour of a human driver.
Training data is collected from recordings human driving in the simulator, then fed into a deep learning network which learns the response (steering angle) for every encountered frame in the simulation. The model is then validated on a new track to check for generalization of the learned features for performing steering angle prediction.*

This project is influenced by [nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), [comma.ai paper](https://arxiv.org/pdf/1608.01230v1.pdf), [vivek's blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.5dpi87xzi) and [various posts](https://medium.com/self-driving-cars/6-different-end-to-end-neural-networks-f307fa2904a5#.yk2a9g6ui) from other Udacity students which I consulted while working on my solution. The [Keras Deep Learning library](https://keras.io/) was used with [Tensorflow](https://www.tensorflow.org/) backend to perform deep learning operations. The training was performed on [Amazon EC2 GPU instances](https://aws.amazon.com/ec2/Elastic-GPUs/).

[//]: # (Image References)
[image0]: ./output_images/project_output.gif
[image1]: ./output_images/histogram_udacity_labels.png
[image2]: ./output_images/hog_RGB2YCrCb.png
[image3]: ./output_images/search_area_and_boxes3.jpg
[image4]: ./output_images/labled_boxes3.jpg
[image5]: ./output_images/heat_map3.jpg
[Advanced Lane Lines]: https://github.com/jinchenglee/CarND-Advanced-Lane-Lines 

[link1]: https://jacobgil.github.io/deeplearning/vehicle-steering-angle-visualizations "Blog: Vehicle steering angle visualization"
[link2]: https://arxiv.org/pdf/1512.04150.pdf "Paper: Learning Deep Features for Discriminative Localization"
[link3]: https://arxiv.org/pdf/1610.02391v1.pdf "Paper: Grad-CAM. Visual Explanations from Deep Networks via Gradient-based Localization"
[link4]: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
[link5]: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.2d9nkoc46 "Vivek's blog on image augmentation"


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

## File Structure
* model.py: used to create and train the model
* model.h5: containing a trained convolution neural network 
* drive.py: used for driving the car in autonomous mode
* analyze_data.py: used to analyze the distribution of recorded training data, say histogram of steering angle against speed or throttle.
* README.md: project writeup

To run the Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```python
python3 drive.py model.h5 [recording_dir]
```

If [recording_dir] is specified, frame images will be automatically saved for later analysis.

To convert images to video:
```python
python3 video.py <img_dir>
```

## Data Collection and Analysis

I started with Udacity's data (known good center lane driving data). Here is an example image form left, center and right camera:

![alt text][image1]

Training by using Udacity data alone provided ok result but the car eventaully drifted outside of the lane. This is mainly because the model hasn't learned how to recover when the car is drifted to the side. Hence I added my own recovery data which was collected using Udacity simulator in training mode. For the recoever driving, I let the car drift to the edge of the lane and recorded steering the car back to the center. 

I also noticed that the data provided by Udacity is out of blance. Below is the histogram of the steering angle data, the majority of the steering angles are 0.0 or very small values. The dominance of the small values would impact the training results. 
![alt text][image1]

Adding more receovery data is also helping balancing the steeing angle distribution. 

## Data Augmentation
After collecting data, I have ~20K images to work with. I then preprocessed the images. For example, modifying image brightness histogram and adding random shadows. Here are example images: 

### Brightness Augmentation
I converted camera image's brightness so the car can learn to operate various lighting conditions. To do brightness augmentation, I converted RGB image to HSV, scaled V (brightness) channel by a random number between .25 and 1.25, and converted the image back to RGB.

```python
def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
```
![alt text][image]

### Random Shadow Augmentation
Random shadow augmentation (copied code from Vivek's blog Vivek's blog  [link5]), which helps A LOT for track 2 with various shadows on the track.

```python
def add_random_shadow(image):
    top_y = image.shape[1]*np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1]*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    random_bright = .15+.8*np.random.uniform()
    if np.random.randint(2)==1:
    #    random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image 
```
![alt text][image]

### Random Flipping
This could simulate the car driving in reverse direction. I only flip the image when the steer angle is greater than 0.1. 

```python
        # flip image (randomly)
        if np.random.uniform()>= 0.5 and abs(steer_ang) > 0.1:
            image = cv2.flip(image, 1)
            steer_ang = -1.0*steer_ang
```

### The Pipline

The image preprocess pipeline includes warping, random flipping and random shadow.

## Model Architecture and Training
The overall strategy for deriving a model architecture was to try, error, analyze failures and improve. 

NVidia-model: My first step was to use a convolution neural network model similar to the NVidia end-to-end paper. I thought this model might be a good starting point because it is a proven model that works for real-world road autonomous driving. 

Image cropping: The nvidia network expects image input size of 200x66, and because I believe the upper 1/3-1/4 part of the input image has no meaning to determine my steering angle, I did a cropping of the upper part then scale to the size of 200x66 in model.py. 
``` python
            # crop image region of interest
            image = crop_image(image, 20, 140, 0+trans_range, im_x-trans_range)
            img = cv2.resize(img, (200,66))
```
Train/Validate split: In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

Overfitting: To combat the overfitting, I modified the model by inserting dropouts to layers so that each layer can learn "redundant" features that even some are dropped in dropout layer, it can still predict the right angle. It did work. 

Test: The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (say, the first left turn before black bridge, the left turn after black bridge, and then the right turn after that)... to improve the driving behavior in these cases, I purposely recorded recovery behavior (from curb side to center of the road) along the tracks. Then the car can finish track 1 completely. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

## Further Improvements
- The driving is extremly wobbly. I'd like to continue to investigate the root cause and optimize the preprocessing and the model

- Better understand the intermediate state of the model and the features that being extracted by the nueral network. At the end of my project, I found a very good blog ([link1]) describing the idea of Activation Mapping. The blog itself was referring to papers: [link2] and [link3]. The whole idea is to using heatmap to highlight locality areas contributing most to the final decision. It was designed for classification purpose, but with slight change, it can be applied to our steering angle predictions. 
