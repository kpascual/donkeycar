'''

pilots.py

Methods to create, use, save and load pilots. Pilots 
contain the highlevel logic used to determine the angle
and throttle of a vehicle. Pilots can include one or more 
models to help direct the vehicles motion. 

'''




import os
import numpy as np
from collections import namedtuple

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers.wrappers import TimeDistributed as TD
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, Cropping3D, Conv2DTranspose

from nn.base import KerasPilot

import donkeycar as dk
from donkeycar import utils 

if tf.__version__ == '1.13.1':
    from tensorflow import ConfigProto, Session

    # Override keras session to work around a bug in TF 1.13.1
    # Remove after we upgrade to TF 1.14 / TF 2.x.
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = Session(config=config)
    keras.backend.set_session(session)

def get():
    return KerasKen()

class KerasKen(KerasPilot):
    '''
    A Keras part that take an image and IMU vector as input,
    outputs steering and throttle

    Note: When training, you will need to vectorize the input from the IMU.
    Depending on the names you use for imu records, something like this will work:

    X_keys = ['cam/image_array','imu_array']
    y_keys = ['user/angle', 'user/throttle']
    
    def rt(rec):
        rec['imu_array'] = np.array([ rec['imu/acl_x'], rec['imu/acl_y'], rec['imu/acl_z'],
            rec['imu/gyr_x'], rec['imu/gyr_y'], rec['imu/gyr_z'] ])
        return rec

    kl = KerasIMU()

    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    '''
    def __init__(self, model=None, cfg = None, num_outputs=2, num_imu_inputs=3, input_shape=(120, 160, 3), *args, **kwargs):
        super(KerasKen, self).__init__(*args, **kwargs)
        self.num_imu_inputs = num_imu_inputs
        self.model = default_ken(num_outputs = num_outputs, num_imu_inputs = num_imu_inputs, input_shape=input_shape)
        self.compile()

        image_h = 120
        image_w = 160

        self.source_inputs = {
            'image': {
                'sources': ('cam/image_array', 'tub_path'), 
                'transform': preprocess_image_train, 
                'shape': (120, 160, 3)
            }, 
            'beacons': {
                'sources': ('beacons/beacon1', 'beacons/beacon2', 'beacons/beacon3'), 
                'transform': preprocess_beacons,
                'shape': (3,)
            }, 
        }
        self.production_inputs = ['cam/normalized/cropped', 'beacons/beacon1', 'beacons/beacon2', 'beacons/beacon3']
        self.processed_inputs = []
        self.outputs = ['angle', 'throttle']
        

    # For training
    def reshape_inputs(self, records, batch_size):
        X = []
        for k, i in self.source_inputs.items():
            # 1. Get entire list from inputs
            arr = [r[k] for r in records]

            new_shape = [batch_size]
            new_shape.extend(i['shape'])

            # 2. Transform into numpy object with appropriate shape
            X.append(np.array(arr).reshape(new_shape))
            
        return X


    def preprocess(self, record):

        processed = {}
        for key, i in self.source_inputs.items():
            func_transform = i['transform']
            func_transform_args = i['transform_args'] if 'transform_args' in i else {}

            inputs = {channel: record[channel] for channel in record if channel in i['sources']}
            processed[key] = func_transform(inputs, **func_transform_args)

        return processed


    def get_input_channels_production(self):
        return self.production_inputs()


    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                  loss='mse')
        
    def run(self, img_arr, beacon1, beacon2, beacon3):
        #TODO: would be nice to take a vector input array.
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        imu_arr = np.array([beacon1, beacon2, beacon3]).reshape(1,self.num_imu_inputs)
        outputs = self.model.predict([img_arr, imu_arr])
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


def preprocess_beacons(inputs):
    return np.array(list(inputs.values())).reshape(1, len(inputs))


def preprocess_image_train(inputs):
    
    # Simulate the config file here
    CFG = namedtuple('CFG', ['IMAGE_H', 'IMAGE_W', 'IMAGE_DEPTH', 'ROI_CROP_TOP', 'ROI_CROP_BOTTOM'])
    cfg = CFG(IMAGE_H = 120, IMAGE_W = 160, IMAGE_DEPTH = 3, ROI_CROP_TOP = 0, ROI_CROP_BOTTOM = 0)
    
    full_path = inputs['tub_path'] + '/' + inputs['cam/image_array']
    return utils.load_scaled_image_arr(full_path, cfg)


def preprocess_image_production(inputs):

    # Simulate the config file here
    CFG = namedtuple('CFG', ['ROI_CROP_TOP', 'ROI_CROP_BOTTOM'])
    cfg = CFG(ROI_CROP_TOP = 0, ROI_CROP_BOTTOM = 0)
    utils.normalize_and_crop(img_arr, cfg)
    


def preprocess_camera(inputs, record):
    # Just 1 value
    i = inputs[0]

    # Get filename

# X parts: camera, beacons
# Y: throttle, angle
def default_ken(num_outputs, num_imu_inputs, input_shape):

    #we now expect that cropping done elsewhere. we will adjust our expeected image size here:
    #input_shape = adjust_input_shape(input_shape, roi_crop)

    img_in = Input(shape=input_shape, name='img_in')
    imu_in = Input(shape=(num_imu_inputs,), name="imu_in")
    
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    
    y = imu_in
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    
    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    outputs = [] 
    
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))
        
    model = Model(inputs=[img_in, imu_in], outputs=outputs)
    
    return model

