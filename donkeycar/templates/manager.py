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

import donkeycar as dk


vthread = None

class ManagerHandler(tornado.web.RequestHandler):
    def get(self):
        drivers = os.listdir('drivers')
        carconfigs = os.listdir('carconfigs')
        self.render("drives/webui/home.html", drivers=drivers, carconfigs=carconfigs)

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


class ConfigHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("drives/webui/config.html", config = cfg.__dict__)

    def post(self):
        config = self.get_body_argument("config")
        path = self.get_body_argument("path")
        print("config: ")
        print(config)
        print("path: ")
        print(path)
        configure(config, path)
        self.write(config)
        


def open_webui():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    static_file_path = os.path.join(this_dir, 'drives/webui', 'static')
    print(static_file_path)

    app = tornado.web.Application([
        ('/', ManagerHandler),
        ('/drive', DriveHandler),
        ('/config', ConfigHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_file_path}),
    ])

    app.listen(8889)
    tornado.ioloop.IOLoop.current().start()


def instantiate_car():
    pass


def check_car():
    V.check_car()


def drive_car():
    V.start(
        rate_hz=cfg.DRIVE_LOOP_HZ, 
        max_loop_count=cfg.MAX_LOOPS
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
        vthread = threading.Thread(target=vehicle.start, args=(cfg.DRIVE_LOOP_HZ, cfg.MAX_LOOPS))
        vthread.is_on = True
        vthread.start()


def stop():
    global vthread
    vthread.is_on = False
    vthread.join()
    vthread = None
    print("thread stopped")


def configure(config, path= None):
    oldconfig = dk.load_config('carconfigs/' + config)
    parts, oldCFG = oldconfig.MAIN()

    remove_existing_parts()
    add_parts(parts)
    vehicle.set_config(path = path)

def remove_existing_parts():
    for p in vehicle.parts:
        vehicle.remove(p)

    vehicle.channels = []


def add_parts(parts):
    # Add parts

    ordered_parts = instantiate(parts)
    for part in ordered_parts:
        vehicle.add_part(part['part'], inputs=part['inputs'], outputs=part['outputs'], threaded=part['threaded'])


class toobj(object):
    def __init__(self, d):
        self.__dict__ = d


def instantiate(parts):
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

def main():
    configfile = 'defaults.yml'
    dk.drive_manager.main(configfile)

if __name__ == '__main__':
    main()



