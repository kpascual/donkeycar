#!/usr/bin/env python3
"""
Vehicle configuration file to control a car through a Raspberry Pi
"""

from docopt import docopt
import donkeycar as dk

#import parts
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.utils import *
from donkeycar.parts.controller import get_js_controller


cfg = dk.load_config()
CFG = cfg
# 1. power train
steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
steering = PWMSteering(
    controller = steering_controller,
    left_pulse = cfg.STEERING_LEFT_PWM, 
    right_pulse = cfg.STEERING_RIGHT_PWM
)

throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
throttle = PWMThrottle(
    controller = throttle_controller,
    max_pulse = cfg.THROTTLE_FORWARD_PWM,
    zero_pulse = cfg.THROTTLE_STOPPED_PWM, 
    min_pulse = cfg.THROTTLE_REVERSE_PWM
)

# 2. driver
drivers = []

ctr = get_js_controller(cfg)



# 3. sensors
from donkeycar.parts.camera import PiCamera
cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)

V = dk.vehicle.Vehicle(
  cfg
)

# Order matters: 1) output throttle & steering 2) anything that interacts with throttle/angle 3) parts that output (but not input) data 4) anything that depends on another part
V.add(
    ctr, 
    inputs=[],
    outputs=['angle', 'throttle', 'user/mode', 'recording'],
    threaded=True
)
V.add(throttle, inputs=['throttle'], outputs=[])
V.add(steering, inputs=['angle'], outputs=[])
V.add(cam, outputs=['cam/image_array'], threaded=threaded)
 




