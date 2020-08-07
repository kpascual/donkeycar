#!/usr/bin/env python3
"""
Vehicle configuration file to control a car through a Raspberry Pi
"""

from docopt import docopt
import donkeycar as dk
import yaml

#import parts
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.utils import *
from donkeycar.parts.controller import get_js_controller2
from donkeycar.parts.camera import PiCamera



def MAIN(configfile = 'defaults.yml', driver_name = None):
    cfg = yaml.load(open(configfile, 'r'))

    parts = []

    # 1. power train
    print(cfg['PCA9685_I2C_ADDR'])
    print(type(cfg['PCA9685_I2C_ADDR']))
    print(cfg['PCA9685_I2C_BUSNUM'])
    print(type(cfg['PCA9685_I2C_BUSNUM']))
    
    steering_controller = PCA9685(cfg['STEERING_CHANNEL'], cfg['PCA9685_I2C_ADDR'], busnum=cfg['PCA9685_I2C_BUSNUM'])
    steering = PWMSteering(
        controller = steering_controller,
        left_pulse = cfg['STEERING_LEFT_PWM'], 
        right_pulse = cfg['STEERING_RIGHT_PWM']
    )

    throttle_controller = PCA9685(cfg['THROTTLE_CHANNEL'], cfg['PCA9685_I2C_ADDR'], busnum=cfg['PCA9685_I2C_BUSNUM'])
    throttle = PWMThrottle(
        controller = throttle_controller,
        max_pulse = cfg['THROTTLE_FORWARD_PWM'],
        zero_pulse = cfg['THROTTLE_STOPPED_PWM'], 
        min_pulse = cfg['THROTTLE_REVERSE_PWM']
    )

    # 2. driver
    drivers = []

    ctr = get_js_controller2(
        controller_type = cfg['CONTROLLER_TYPE'],
        joystick_throttle_dir = cfg['JOYSTICK_THROTTLE_DIR'], 
        joystick_max_throttle = cfg['JOYSTICK_MAX_THROTTLE'], 
        joystick_steering_scale = cfg['JOYSTICK_STEERING_SCALE'], 
        auto_record_on_throttle = cfg['AUTO_RECORD_ON_THROTTLE'], 
        joystick_deadzone = cfg['JOYSTICK_DEADZONE']
    )


    # 3. sensors
    cam = PiCamera(image_w=cfg['IMAGE_W'], image_h=cfg['IMAGE_H'], image_d=cfg['IMAGE_DEPTH'])

    # 4. vehicle run configurations


    #### SHOULD END HERE

    parts = [
        {
            'part': throttle, 
            'inputs': ['throttle'], 
            'outputs': [], 
            'threaded': False
        },
        {
            'part': steering, 
            'inputs': ['angle'], 
            'outputs': [], 
            'threaded': False
        },
        {
            'part': cam, 
            'inputs': [], 
            'outputs': ['cam/image_array'], 
            'threaded': True
        },
        {
            'part': ctr, 
            'inputs': [], 
            'outputs': ['angle', 'throttle', 'user/mode', 'recording'],
            'threaded': True
        },
    ]
    channels = [
        ('throttle', 'float'),
        ('angle', 'float'),
        ('cam/image_array', 'image_array'),
        ('user/mode', 'str'),
        ('recording', 'boolean'),
    ]
 
    return parts, channels



