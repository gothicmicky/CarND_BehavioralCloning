import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version
from img_preprocess import img_preprocess

import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def preprocess(image):
    # get shape and chop off 1/3 from the top
    shape = image.shape
    print("shape: ",shape)
    # note: numpy arrays are (row, col)!
    image = image[40:shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)
    # image = cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)
    return image

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))

        image_pre = np.asarray(image)
        image_array = preprocess(image_pre)
        # Crop unnecessary image top lines
        # img_crop = image_array[56:160,:,:]
        # img_resize = cv2.resize(img_crop, (200,66))
        # img_normed = normalize_color(img_resize[None,:,:,:])

        transformed_image_array = image_array[None, :, :, :]
        print(transformed_image_array.shape)

        # This model currently assumes that the features of the model are just the images. Feel free to change this.
        steering_angle = 1.0*float(model.predict(transformed_image_array, batch_size=1))
        
        # Speed limits control
        min_speed = 10
        max_speed = 20 
        if float(speed) < min_speed:
            throttle = 3.0
        elif float(speed) > max_speed:
            throttle = -1.0
        else:
            throttle = 3.0
        
        # #Adaptive throttle - Both Track
        # if (abs(float(speed)) < 10):
        #     throttle = 0.5
        # else:
        #     # When speed is below 20 then increase throttle by speed_factor
        #     if (abs(float(speed)) < 25):
        #         speed_factor = 1.35
        #     else:
        #         speed_factor = 1.0

        #     if (abs(steering_angle) < 0.1): 
        #         throttle = 0.3 * speed_factor
        #     elif (abs(steering_angle) < 0.5):
        #         throttle = 0.2 * speed_factor
        #     else:
        #         throttle = 0.15 * speed_factor

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            # Recorded image with right color as using OpenCV (BGR).
            img_record = cv2.cvtColor(img_resize, cv2.COLOR_RGB2BGR)
            #print("saving...")

            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)

            # Save original image?
            #image.save('{}.png'.format(image_filename))

            # Draw a line on image to show steering direction
            if args.draw_steer == 'on':
                start_x = 100
                start_y = 66
                mid_x = 100
                mid_y = 60
                end_y = 30
                end_x = start_x + steering_angle * (start_y - end_y)
                end_x = int(end_x)

                points_list = []
                points_list.append((start_x, start_y))
                points_list.append((mid_x, mid_y))
                points_list.append((end_x, end_y))
                cv2.polylines(img_record, [np.array(points_list)], False, (255,0,0), thickness=1, lineType=cv2.LINE_AA)

            # Add steering angle to file name.
            image_filename = image_filename + '_' + str(steering_angle)
            # Save angle-annotated image?
            cv2.imwrite('{}.png'.format(image_filename), img_record)
 
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    parser.add_argument(
        'draw_steer',
        type=str,
        nargs='?',
        default='on',
        help='Want to draw the steering angle on image? On/off. Default on.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
            ', but the model was built using ', model_version)
        
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
