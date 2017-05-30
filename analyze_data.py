import os
import pickle
import json
import random
import csv

import cv2
import numpy as np
import os
import csv
import glob

import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from img_preprocess import img_preprocess, augment_brightness
data_dir = 'data2'

dirs = os.listdir(data_dir)
print(dirs)
####################################################################

# Load Udacity's data - images are in IMG_udacity
# df = pd.read_csv(data_dir + '/'+'driving_log.csv', header=0)
# df.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]
# print('normal driving data')
# print(len(df))
# #print(df[0:3])

# # Load Generate Recovery Data - images are in IMG_recovery
# df_recovery = pd.read_csv(data_dir+'driving_log_recovery.csv', header=0)
# df_recovery.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]
# print('recoveriy driving data')
# print(len(df_recovery))
# #print(df_recovery[0:3])

frames = []
# load training data
for dir in dirs:
	df = pd.read_csv(data_dir + '/' + dir + '/driving_log.csv', header=0)
	df.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]
	print('dir: ', dir)
	print('len: ', len(df))
	frames.append(df)

pd_total = pd.concat(frames)

print('len: ', len(pd_total))


####################################################################

# Visualize left, center and right angle camera at the same moment
chk_idx = int(np.random.uniform(1, len(df)))
#print(chk_idx)
img_left = cv2.imread(data_dir+df["left_image"][chk_idx].strip())
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_center = cv2.imread(data_dir+df["center_image"][chk_idx].strip())
img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
img_right = cv2.imread(data_dir+df["right_image"][chk_idx].strip())
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
angle = df["steering_angle"][chk_idx]

row, col, ch = img_center.shape
img_center = img_center.reshape(row, col, ch)
img_center_processed, angle_processed = img_preprocess(img_center, angle)
img_center_resized = cv2.resize(img_center_processed, (200,66), interpolation=cv2.INTER_AREA)


plt.rcParams["figure.figsize"] = [32, 24]
plt.tick_params(axis='x', labelsize=25)
plt.subplot(1, 3, 1)
plt.imshow(img_center)
plt.title("ORIGINAL: " + str(np.round(angle, 2)), fontsize=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

plt.subplot(1, 3, 2)
plt.imshow(img_center_processed)
plt.title("Processed: " + str(np.round(angle_processed, 2)), fontsize=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

plt.subplot(1, 3, 3)
plt.imshow(img_center_resized)
plt.title("Resized: " + str(np.round(angle_processed, 2)), fontsize=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

plt.tight_layout()
plt.show()

# plt.rcParams["figure.figsize"] = [32, 24]
# plt.tick_params(axis='x', labelsize=25)

# plt.subplot(1, 3, 1)
# plt.imshow(img_left)
# plt.title("LEFT: " + str(np.round(angle, 2)), fontsize=20)
# plt.tick_params(axis='x', labelsize=20)
# plt.tick_params(axis='y', labelsize=20)

# plt.subplot(1, 3, 2)
# plt.imshow(img_center)
# plt.title("CENTER: " + str(np.round(angle, 2)), fontsize=20)
# plt.tick_params(axis='x', labelsize=20)
# plt.tick_params(axis='y', labelsize=20)

# plt.subplot(1, 3, 3)
# plt.imshow(img_right)
# plt.title("RIGHT: " + str(np.round(angle, 2)), fontsize=20)
# plt.tick_params(axis='x', labelsize=20)
# plt.tick_params(axis='y', labelsize=20)

# plt.tight_layout()
# plt.show()

# # for i in range(3):
# #     print (i, '{:}'.format(data_dir+ df["center_image"][i]))

# ####################################################################
# # This is a 160 pixel x 320 pixel x 3 channels
# IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = img_center.shape
# print("img_height:", IMG_HEIGHT)
# print("img_width:", IMG_WIDTH)
# print("img_channel:", IMG_CHANNEL)

# # plt.figure(figsize=(4, 3))
# # plt.tick_params(axis='x', labelsize=10)
# # plt.tick_params(axis='y', labelsize=10)
# # plt.imshow(img_center)

# min_angle = np.min(df["steering_angle"])
# max_angle = np.max(df["steering_angle"])
# print(min_angle, max_angle)

# # Time Series plot of steering angles
# plt.figure(figsize=(6, 4))
# ts = df["steering_angle"]
# ts.plot()

# # Histogram plot of steering angles
# plt.figure(figsize=(8,3))
# plt.hist(ts.astype('float'), bins=np.arange(-1.0, 1.0, 0.01))
# plt.title("Steering Angle Distribution - normal driving")
# plt.show()


# ts_recoevery = df_recovery["steering_angle"]
# # Histogram plot of steering angles
# plt.figure(figsize=(8,3))
# plt.hist(ts_recoevery.astype('float'), bins=np.arange(-1.0, 1.0, 0.01))
# plt.title("Steering Angle Distribution - recovery driving")
# plt.show()

# ####################################################################
# # Data augmentation (upsampling and downsampling)
# # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# df_right = []
# df_left = []
# df_center = []
# for i in range(len(df)):
#     center_img = df["center_image"][i]
#     left_img = df["left_image"][i]
#     right_img = df["right_image"][i]
#     angle = df["steering_angle"][i]
    
#     if (angle > 0.15):
#         df_right.append([center_img, left_img, right_img, angle])
        
#         # I'm adding a small deviation of the angle 
#         # This is to create more right turning samples for the same image
#         for i in range(10):
#             new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
#             df_right.append([center_img, left_img, right_img, angle])
            
#     elif (angle < -0.15):
#         df_left.append([center_img, left_img, right_img, angle])
        
#         # Adding a small deviation of the angle
#         # This is to create more left turning samples for the same image
#         for i in range(5):
#             new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
#             df_left.append([center_img, left_img, right_img, new_angle])
            
#     else:
#         if (angle != 0.0):
#             # Include all near 0 angle data
#             df_center.append([center_img, left_img, right_img, angle])


# print(len(df_center), len(df_left), len(df_right))

# # process recovery image
# for i in range(len(df_recovery)):
#     center_img = df_recovery["center_image"][i]
#     left_img = df_recovery["left_image"][i]
#     right_img = df_recovery["right_image"][i]
#     angle = df_recovery["steering_angle"][i]
    
#     if (angle > 0.15):
#         df_right.append([center_img, left_img, right_img, angle])
        
#         # I'm adding a small deviation of the angle 
#         # This is to create more right recovery samples for the same image
#         for i in range(10):
#             new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
#             df_right.append([center_img, left_img, right_img, angle])
            
#     elif (angle < -0.15):
#         df_left.append([center_img, left_img, right_img, angle])
        
#         # I'm adding a small deviation of the angle
#         # This is to create more left recovery samples for the same image
#         for i in range(7):
#             new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
#             df_left.append([center_img, left_img, right_img, new_angle])
            
#     else:
#         # for i in range(5):
#         #     new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
#         #     df_center.append([center_img, left_img, right_img, new_angle])
#         if (angle != 0.0):
#             df_center.append([center_img, left_img, right_img, angle])

# print(len(df_center), len(df_left), len(df_right))

# # Shuffle the data so they're no longer sequential in the order that the data was collected
# random.shuffle(df_center)
# random.shuffle(df_left)
# random.shuffle(df_right)

# df_center = pd.DataFrame(df_center, columns=["center_image", "left_image", "right_image", "steering_angle"])
# df_left = pd.DataFrame(df_left, columns=["center_image", "left_image", "right_image", "steering_angle"])
# df_right = pd.DataFrame(df_right, columns=["center_image", "left_image", "right_image", "steering_angle"])

# ####################################################################
# # Make the train and valid list 
# data_list = [df_center, df_left, df_right]
# data_list_df = pd.concat(data_list, ignore_index=True)

# print('length of final data list')
# print(len(data_list_df))

# ts_cancat = data_list_df["steering_angle"]
# # Histogram plot of steering angles
# plt.figure(figsize=(8,3))
# plt.hist(ts_cancat.astype('float'), bins=np.arange(-1.0, 1.0, 0.01))
# plt.title("Steering Angle Distribution - final list")
# plt.show()
