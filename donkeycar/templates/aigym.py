#!/usr/bin/env python3
"""
Vehicle configuration file to control a car through a Raspberry Pi
"""
import os
import yaml
import donkeycar as dk

from donkeycar.parts.controller import get_js_controller2
from donkeycar.parts.dgym import DonkeyGymEnv
from donkeycar.parts.network import MQTTValuePub


def MAIN(configfile = 'defaults.yml', driver_name = None):
    parts = []
    newcfg = yaml.load(open(configfile, 'r'))

    # ensure we don't run out of resources using cuda
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    # 1. power train
    # No drive train on the gym

    # 2. driver
    #drivers = []
    ctr = get_js_controller2(
        controller_type = newcfg['CONTROLLER_TYPE'],
        joystick_throttle_dir = newcfg['JOYSTICK_THROTTLE_DIR'], 
        joystick_max_throttle = newcfg['JOYSTICK_MAX_THROTTLE'], 
        joystick_steering_scale = newcfg['JOYSTICK_STEERING_SCALE'], 
        auto_record_on_throttle = newcfg['AUTO_RECORD_ON_THROTTLE'], 
        joystick_deadzone = newcfg['JOYSTICK_DEADZONE']
    )

    # 3. sensors
    print(newcfg['DONKEY_SIM_PATH'])
    print(newcfg['SIM_ARTIFICIAL_LATENCY'])
    cam = DonkeyGymEnv(
        sim_path = newcfg['DONKEY_SIM_PATH'], 
        host     = newcfg['SIM_HOST'], 
        env_name = newcfg['DONKEY_GYM_ENV_NAME'], 
        conf     = newcfg['GYM_CONF'], 
        delay    = newcfg['SIM_ARTIFICIAL_LATENCY']
    )
    telemetry_throttle = MQTTValuePub(name="jumanjijenkins/donkeytest/throttle", broker="localhost")
    telemetry_steering = MQTTValuePub(name="jumanjijenkins/donkeytest/steering", broker="localhost")

    # 4. vehicle run configurations


    #### SHOULD END HERE


    parts = [
        {
            'part': cam, 
            'inputs': ['angle', 'throttle'], 
            'outputs': ['cam/image_array','speed', 'pos'], 
            'threaded': True
        },
        {
            'part': ctr, 
            'inputs': [], 
            'outputs': ['angle', 'throttle', 'user/mode', 'recording'],
            'threaded': True
        },
        {
            'part': telemetry_throttle, 
            'inputs': ['throttle'], 
            'outputs': [], 
            'threaded': False
        },
        {
            'part': telemetry_steering, 
            'inputs': ['angle'], 
            'outputs': [], 
            'threaded': False
        },
    ]

    channels = [
        ('cam/image_array', 'image_array'),
        ('angle', 'float'),
        ('throttle', 'float'),
        ('speed', 'float'),
        ('pos', 'vector'),
        ('user/mode', 'str'),
        ('recording', 'boolean'),
    ]
 
    return parts, channels



