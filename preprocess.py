import os
import pickle
import json
import random
import csv

import cv2
import numpy as np

####################################################################

def crop_image(image, y1, y2, x1, x2):
    """
    crop image into respective size
    give: the crop extent
    """
    return image[y1:y2, x1:x2]

####################################################################

def flip_image(image, angle):
    img_flip = cv2.flip(image,1)
    angle = -angle
        
    return img_flip, angle

####################################################################

def warp_img(img, angle):
    '''
    Warp image horizontally, calculate the angle shifted then append to orig angle.
    '''
    WARP_DIV_RATIO = 5

    rows,cols,ch = img.shape
    
    # shifts within 1/(WARP_DIV_RATIO) of image width
    shifted_pixel = random.randint(-1*cols//WARP_DIV_RATIO,cols//WARP_DIV_RATIO)
    #print(shifted_pixel)
    
    pts1 = np.float32([[cols//2,0],[0,rows-1],[cols-1,rows-1]])
    pts2 = np.float32([[cols//2+shifted_pixel,0],[0,rows-1],[cols-1,rows-1]])
    
    delta_angle = 0.004*shifted_pixel
    total_angle = angle + delta_angle
    #print(delta_angle, total_angle)
    
    M = cv2.getAffineTransform(pts1,pts2)
    warp_img = cv2.warpAffine(img,M,(cols,rows))
    
    return warp_img, total_angle

####################################################################

def augment_brightness(image):
    """
    apply random brightness on the image
    """
    scale_factor = .25
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = scale_factor+np.random.uniform()
    
    # scaling up or down the V channel of HSV
    image[:,:,2] = image[:,:,2]*random_bright

    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

####################################################################

# Below func. copied from: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.kgjn97cup
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

####################################################################

def change_brightness(image):
    # Randomly select a percent change
    change_pct = random.uniform(0.4, 1.2)
    
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    
    #Convert back to RGB 
    img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_bright

####################################################################

def trans_image(image,steer,trans_range, trans_y=False):
    """
    translate image and compensate for the translation on the steering angle
    """
    rows, cols, chan = image.shape
    
    # horizontal translation with 0.008 steering compensation per pixel
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*.4
    
    # option to disable vertical translation (vertical translation not necessary)
    if trans_y:
        tr_y = 40*np.random.uniform()-40/2
    else:
        tr_y = 0
    
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

def affine_transform(img, angle, pixels, angle_adjust, right=True):

    cols, rows, ch = img.shape
        
    pts1 = np.float32([[10,10], [200,50], [50,250]])

    if right:
        pts2 = np.float32([[10, 10], [200+pixels, 50], [50, 250]])
        angle =- angle_adjust
    else:
        pts2 = np.float32([[10, 10], [200-pixels, 50], [50, 250]])
        angle =- angle_adjust

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (rows, cols))

    return dst.reshape((cols, rows, ch)), angle


def img_preprocess(image, steer_ang, flip = True, warp = True, shadow = True):
    """
    Apply processing to image: The input of the image is BGR format
    """    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # image size
    im_y, im_x, ch = image.shape
    
    # translate image and compensate for steering angle
    trans_range = 0
    # image, steer_ang = trans_image(image, steer_ang, trans_range) # , trans_y=True
    
    # crop image region of interest
    #image = crop_image(image, 20, 140, 0+trans_range, im_x-trans_range)
    image = crop_image(image, 56, 140, 0+trans_range, im_x-trans_range)
       
    # Coin flip to see to flip image and create a new sample of -angle
    if flip and steer_ang != 0 and np.random.randint(2) == 1:
        image, steer_ang = flip_image(image, steer_ang)

    # augment brightness
    #image = augment_brightness(image)
    #image = change_brightness(image)

    if shadow:
        image = add_random_shadow(image)

    if warp and steer_ang == 0:
        image, steer_ang = warp_img(image, steer_ang)

    # perturb steering with a bias
    #steer_ang += np.random.normal(loc=0,scale=0.2)
        
    return image, steer_ang
