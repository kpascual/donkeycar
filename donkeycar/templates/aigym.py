#!/usr/bin/env python3
"""
Vehicle configuration file to control a car through a Raspberry Pi
"""
import os
from docopt import docopt
import yaml
import donkeycar as dk

#import parts
from donkeycar.parts.controller import get_js_controller
from donkeycar.parts.dgym import DonkeyGymEnv
from donkeycar.parts.network import MQTTValuePub
from tensorflow.python import keras


def MAIN(configfile = 'defaults.yml', model_path = None):
    cfg = dk.load_config()
    CFG = cfg

    parts = []
    newcfg = yaml.load(open(configfile, 'r'))
    print(newcfg)

    


    # ensure we don't run out of resources using cuda
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    

    # 1. power train
    # No drive train on the gym

    # 2. driver
    #drivers = []
    ctr = get_js_controller(cfg)

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
            'outputs': ['cam/image_array'], 
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
 
    return parts, CFG



