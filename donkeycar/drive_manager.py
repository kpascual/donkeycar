#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manager.py 

Options:
    -h --help          Show this screen.
"""
import threading
import os
import glob
from docopt import docopt
import tornado.ioloop
import tornado.web
import yaml
import json

import donkeycar as dk


vthread = None
vehicle = None
cfg = None
this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(this_dir, "management/drive_manager")

class ManagerHandler(tornado.web.RequestHandler):
    def get(self):
        drivers = [f.name for f in os.scandir('drivers') if f.is_dir() and f.name != '__pycache__']
        carconfigs = os.listdir('carconfigs')
        print(root_dir)
        self.render("home.html", drivers=drivers, carconfigs=carconfigs)

class DriveHandler(tornado.web.RequestHandler):
    def post(self):
        print(self.request.body)
        command = self.get_body_argument("command")
        if command == "start":
            print("start car")
            driver = self.get_body_argument("driver")
            self.write("you started car: "  + driver)
            start()
        elif command == "stop":
            print("stop car")
            self.write("you stopped car")
            stop()


class CarConfigHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("config.html", config = cfg)

    def post(self):
        config = self.get_body_argument("config")
        driver = self.get_body_argument("driver")
        print("config: ")
        print(config)
        if driver == 'manual':
            driver = None
        print(driver)
        parts, channels = configure_car(config, driver)

        return_dict = {
            'parts': [],
            'channels': channels
        }
        return_dict['parts'] = document_parts(parts)
        
        self.write(return_dict)

        
class RecorderConfigHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("config.html", config = cfg)

    def post(self):
        args = json.loads(self.request.body.decode('utf-8'))
        path = args['path']

        print("path: ")
        print(path)

        print(args['channels'])
        channels = args['channels']
        print("channels: ")
        print(channels)

        configure_recorder(channels, path)

        self.write('asdklfjd')



def open_webui():
    static_file_path = os.path.join(root_dir, 'static')

    app = tornado.web.Application([
        ('/', ManagerHandler),
        ('/drive', DriveHandler),
        ('/car_config', CarConfigHandler),
        ('/recorder_config', RecorderConfigHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_file_path}),
    ], template_path = root_dir)

    app.listen(8889)
    tornado.ioloop.IOLoop.current().start()

def instantiate_car(configfile):

    global cfg
    cfg = yaml.load(open(configfile, 'r'), Loader=yaml.Loader)
    #cfg = toobj(cfg)
    
    global vehicle
    vehicle = dk.vehicle.Vehicle(cfg['DATA_PATH'])


def check_car():
    V.check_car()


def drive_car():
    V.start(
        rate_hz=cfg['DRIVE_LOOP_HZ'], 
        max_loop_count=cfg['MAX_LOOPS']
    )

# Vehicle
#   register_driver
#   change_driver
#   add_part
#   add_sensor
#   define_what_to_record
#   start_recording
#   stop_recording

def check(vehicle):
    vehicle.check()


def start():
    global vthread

    if vthread is None:
        vthread = threading.Thread(target=vehicle.start, args=(cfg['DRIVE_LOOP_HZ'], cfg['MAX_LOOPS']))
        vthread.is_on = True
        vthread.start()


def stop():
    global vthread
    vthread.is_on = False
    vthread.join()
    vthread = None
    print("thread stopped")


def configure_car(configname, driver = None):
    carconfig = dk.load_config('carconfigs/' + configname)
    parts, channels = carconfig.MAIN(driver_name = driver)

    # 1. set parts
    remove_existing_parts()
    add_parts(parts)
    channel_names = [c[0] for c in channels] 
    vehicle.set_channels(channels)
    vehicle.set_data_recorder_config(inputs=channel_names)

    return parts, channel_names

def configure_recorder(inputs = None, path = None):
    print("configure recorder")
    print(inputs)
    if inputs is not None:
        vehicle.set_data_recorder_config(inputs=inputs)
    
    if path is not None:
        vehicle.set_data_recorder_config(path = path)

    

def remove_existing_parts():
    for p in vehicle.parts:
        vehicle.remove(p)

    vehicle.channels = []


def add_parts(parts):
    # Add parts

    #ordered_parts = rearrange_parts(parts)
    for part in parts:
        vehicle.add_part(part['part'], inputs=part['inputs'], outputs=part['outputs'], threaded=part['threaded'])
        #print(part)


class toobj(object):
    def __init__(self, d):
        self.__dict__ = d


def rearrange_parts(parts):
    # Determine parts dependency order
    # then create new vehicle object

    #vehicle = dk.vehicle.Vehicle(cfg)
    ordered_parts = []
    # 1. check if part has no inputs
    remainder = parts.copy()
    all_inputs = []
    all_outputs = []

    while remainder:
        next_remainder = []
        for part in remainder:
            deps = set(part['inputs'])
            deps.difference_update(all_outputs)
            if deps:
                next_remainder.append(part)
            else:
                ordered_parts.append(part)
                all_outputs.extend(part['outputs'])
        remainder = next_remainder
        print(remainder)

    return ordered_parts


def document_parts(parts):
    docs = []
    for part in parts:
        docs.append({
            'part': part['part'].__class__.__name__,
            'inputs': part['inputs'],
            'outputs': part['outputs'],
            'threaded': part['threaded']
        })

    return docs


def main(configfile):

    global cfg
    cfg = yaml.load(open(configfile, 'r'), Loader=yaml.Loader)
    
    global vehicle
    vehicle = dk.vehicle.Vehicle(cfg['DATA_PATH'])

    open_webui()


if __name__ == '__main__':
    main()



