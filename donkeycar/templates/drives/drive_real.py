#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    drive_real.py check (--config=<config>) [--driver=<driver>] [--js] [--nn=<nn>]
    drive_real.py drive (--config=<config>) [--driver=<driver>] [--js] [--nn=<nn>] [--webui]


Options:
    -h --help          Show this screen.
"""
import os
from docopt import docopt
import tornado.ioloop
import tornado.web

import donkeycar as dk



class ManagerHandler(tornado.web.RequestHandler):
    def get(self):
        #self.write("first web page")
        self.render("webui/test.html")

class DriveHandler(tornado.web.RequestHandler):
    def get(self):
        #self.write("first web page")
        self.render("webui/vehicle.html")

def open_webui():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    static_file_path = os.path.join(this_dir, 'webui', 'static')
    print(static_file_path)

    app = tornado.web.Application([
        ('/', ManagerHandler),
        ('/drive', DriveHandler),
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


def start(vehicle, cfg):
    vehicle.start(
        rate_hz = cfg.DRIVE_LOOP_HZ,
        max_loop_count = cfg.MAX_LOOPS
    )


if __name__ == '__main__':
    args = docopt(__doc__)

    config_path = args['--config']
    #cfg = dk.load_config(config_path)
    
    driver = args['--driver']
    webui = args['--webui']
    #drive(cfg, driver=driver, use_joystick=args['--js'], model_type=model_type, camera_type=camera_type, meta=args['--meta'])


    if webui:
        open_webui()
    #print(cfg)
    """
    if args['drive']:
        start(cfg.V, cfg.CFG)
    
    if args['check']:
        check(cfg.V)
    """

