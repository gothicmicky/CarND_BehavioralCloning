import csv
import numpy as np
import cv2
import tables
import os
import sys
import random


def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    img_max = np.max(image_data)
    img_min = np.min(image_data)
    a = -0.5
    b = 0.5

    img_normed = a + (b-a)*(image_data - img_min)/(img_max - img_min)
    #print(np.max(img_normed))
    #print(np.min(img_normed))
    return img_normed

def normalize_color(image_data):
    """
    Normalize the image data on per channel basis. 
    """
    img_normed_color = np.zeros_like(image_data, dtype=float)
    for ch in range(image_data.shape[2]):
        tmp = normalize_grayscale(image_data[:,:,ch])
        img_normed_color[:,:,ch] = tmp
    #print(np.max(img_normed_color))
    #print(np.min(img_normed_color))
    return img_normed_color

def lower_luma(image):
    RATIO = 0.5
    cv2.imwrite("ori.png", image)
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
    image1[:,:,0] = RATIO*image1[:,:,0]
    image1 = cv2.cvtColor(image1,cv2.COLOR_YUV2RGB)
    cv2.imwrite("after.png", image1)
    return image1

def augment_brightness(image):
    #cv2.imwrite("ori.png", image)
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    #cv2.imwrite("after.png", image1)
    return image1

def darker_img(image):
    # Convert to YUV
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_gray = img_yuv[:,:,0]
   
    # Pick the majority pixels of the image
    idx = (img_gray<245) & (img_gray > 10)
    
    # Make the image darker
    img_gray_scale = img_gray[idx]*np.random.uniform(0.1,0.6)
    img_gray[idx] = img_gray_scale
    
    # Convert back to BGR 
    img_yuv[:,:,0] = img_gray
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

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
    #cv2.imwrite('test.png', img)
    #cv2.imwrite('test_warp.png', warp_img)
    return warp_img, total_angle

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


def crop_image(image, y1, y2, x1, x2):
    """
    crop image into respective size
    give: the crop extent
    """
    return image[y1:y2, x1:x2]

def img_preprocess(image, steer_ang, train=True):
    """
    Apply processing to image
    """    
    # image size
    im_y = image.shape[0]
    im_x = image.shape[1]
    
    # translate image and compensate for steering angle
    trans_range = 50
    # image, steer_ang = trans_image(image, steer_ang, trans_range) # , trans_y=True
    
    # if np.random.uniform()>= 0.5: #and abs(steer_ang) > 0.1
    image, steer_ang= warp_img(image, steer_ang)

    # crop image region of interest
    image = crop_image(image, 20, 140, 0+trans_range, im_x-trans_range)
    
    # flip image (randomly)
    #if np.random.uniform()>= 0.5: #and abs(steer_ang) > 0.1
    image = cv2.flip(image, 1)
    steer_ang = -steer_ang

    # augment brightness
    image = augment_brightness(image)
    
    # perturb steering with a bias
    # steer_ang += np.random.normal(loc=0,scale=0.2)
        
    return image, steer_ang