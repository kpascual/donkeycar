#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 10:44:24 2017

@author: wroscoe
"""

import time
from statistics import median
from threading import Thread
import threading
from .memory import Memory
from prettytable import PrettyTable
from donkeycar.parts.datastore import TubHandler

# How quickly do certain parts run? (e.g. is the drive loop efficient?)
# This is different than telemetry data outputted by the parts
class PartProfiler:
    def __init__(self):
        self.records = {}

    def profile_part(self, p):
        self.records[p] = { "times" : [] }

    def on_part_start(self, p):
        self.records[p]['times'].append(time.time())

    def on_part_finished(self, p):
        now = time.time()
        prev = self.records[p]['times'][-1]
        delta = now - prev
        thresh = 0.000001
        if delta < thresh or delta > 100000.0:
            delta = thresh
        self.records[p]['times'][-1] = delta

    def report(self):
        print("Part Profile Summary: (times in ms)")
        pt = PrettyTable()
        pt.field_names = ["part", "max", "min", "avg", "median"]
        for p, val in self.records.items():
            # remove first and last entry because you there could be one-off
            # time spent in initialisations, and the latest diff could be
            # incomplete because of user keyboard interrupt
            arr = val['times'][1:-1]
            if len(arr) == 0:
                continue
            pt.add_row([p.__class__.__name__,
                        "%.2f" % (max(arr) * 1000),
                        "%.2f" % (min(arr) * 1000),
                        "%.2f" % (sum(arr) / len(arr) * 1000),
                        "%.2f" % (median(arr) * 1000)])
        print(pt)


# Rename Telemetry to TelemetryRecorder
# Rename Memory to Telemetry
# responsibilities: 1) where to save it 2) what info to save
class Telemetry:
    def __init__(self, path):
        print(path)
        self.th = TubHandler(path=path)
        self.mem = Memory()

        self.inputs=[
            'cam/image_array',
            'user/angle', 
            'user/throttle', 
            'angle',
            'throttle',
            'user/mode',
            'beacons/beacon1',
            'beacons/beacon2',
            'beacons/beacon3',
        ]
        self.types=[
            'image_array',
            'float', 
            'float',
            'float', 
            'float',
            'str',
            'int',
            'int',
            'int'
        ]
        self.tub = None


    def create_tub(self, path = None):
        print("Path found:" + path)
        self.tub = self.th.new_tub_writer(inputs=self.inputs, types=self.types, path=path)


    def save_vehicle_configuration(self, parts):
        newparts = []
        for p in parts:
            newparts.append({k:v for k,v in p.items() if k != 'thread' and k != 'part'})

        self.tub.parts = newparts

    def save_end_time(self):
        self.tub.end_time = time.time()


    def record(self):
        inputs = self.mem.get(self.inputs)
        self.tub.run(*inputs)


    def get(self, keys):
        return self.mem.get(keys)

    def put(self, keys, inputs):
        self.mem.put(keys, inputs)

    def cleanup_postsession(self):
        self.save_end_time()
        self.tub.write_meta()


class Vehicle:
    def __init__(self, cfg, mem=None):

        if not mem:
            mem = Memory()
        self.mem = mem
        self.parts = []
        self.driver = None
        self.on = True
        self.threads = []
        self.profiler = PartProfiler()
        self.cfg = cfg
        self.telemetry = Telemetry(cfg.DATA_PATH)
        self.channels = []

        # States that used to be in mem
        self.is_recording = False
        self.is_ai_running = False
        self.driver = 'user' # user | local_angle | local
        # self.driver = Driver()
        # def change_driver
        # def start_recording

    def change_driver(self):
        pass

    def set_config(self, path = None):
        self.telemetry.create_tub(path)

    def add_part(self, part, inputs=[], outputs=[], threaded=False, run_condition=None):
        self.add(part, inputs, outputs, threaded=threaded, run_condition=run_condition)

    def add(self, part, inputs=[], outputs=[],
            threaded=False, run_condition=None):
        """
        Method to add a part to the vehicle drive loop.

        Parameters
        ----------
            inputs : list
                Channel names to get from memory.
            outputs : list
                Channel names to save to memory.
            threaded : boolean
                If a part should be run in a separate thread.
            run_condition : boolean
                If a part should be run or not
        """
        assert type(inputs) is list, "inputs is not a list: %r" % inputs
        assert type(outputs) is list, "outputs is not a list: %r" % outputs
        assert type(threaded) is bool, "threaded is not a boolean: %r" % threaded

        p = part
        print('Adding part {}.'.format(p.__class__.__name__))
        entry = {}
        entry['part'] = p
        entry['part_name'] = p.__class__.__name__
        entry['inputs'] = inputs
        entry['outputs'] = outputs
        entry['run_condition'] = run_condition
        entry['threaded'] = threaded


        self.parts.append(entry)
        self.profiler.profile_part(part)

        self.channels.extend([i for i in inputs if i not in self.channels])
        self.channels.extend([o for o in outputs if o not in self.channels])

    def remove(self, part):
        """
        remove part form list
        """
        self.parts.remove(part)

    def start(self, rate_hz=10, max_loop_count=None, verbose=False):
        """
        Start vehicle's main drive loop.

        This is the main thread of the vehicle. It starts all the new
        threads for the threaded parts then starts an infinite loop
        that runs each part and updates the memory.

        Parameters
        ----------

        rate_hz : int
            The max frequency that the drive loop should run. The actual
            frequency may be less than this if there are many blocking parts.
        max_loop_count : int
            Maximum number of loops the drive loop should execute. This is
            used for testing that all the parts of the vehicle work.
        """
        ct = threading.currentThread()

        try:

            self.on = True

            for entry in self.parts:
                if entry.get('threaded'):
                    # start the update thread
                    t = Thread(target=entry['part'].update, args=())
                    t.daemon = True
                    entry['thread'] = t

                    print("adding thread for part " + entry['part_name'])
                    entry.get('thread').start()
                if hasattr(entry['part'], 'running'):
                    entry['part'].running = True

            # Pre-start checking
            self.telemetry.save_vehicle_configuration(self.parts)

            # wait until the parts warm up.
            print('Starting vehicle...')

            loop_count = 0
            while self.on and getattr(ct, "is_on", True):
                start_time = time.time()
                loop_count += 1

                # update all third party sensors
                self.update_parts()

                # record telemetry
                if self.telemetry.get(['recording'])[0]:
                    self.telemetry.record()

                # then update driver

                # stop drive loop if loop_count exceeds max_loopcount
                if max_loop_count and loop_count > max_loop_count:
                    self.on = False

                sleep_time = 1.0 / rate_hz - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    # print a message when could not maintain loop rate.
                    if verbose:
                        print('WARN::Vehicle: jitter violation in vehicle loop '
                              'with {0:4.0f}ms'.format(abs(1000 * sleep_time)))

                if verbose and loop_count % 200 == 0:
                    self.profiler.report()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def update_parts(self):
        '''
        loop over all parts
        '''
        for entry in self.parts:

            run = True
            # check run condition, if it exists
            if entry.get('run_condition'):
                run_condition = entry.get('run_condition')
                run = self.telemetry.get([run_condition])[0]
            
            if run:
                # get part
                p = entry['part']
                # start timing part run
                self.profiler.on_part_start(p)
                # get inputs from memory
                inputs = self.telemetry.get(entry['inputs'])
                # run the part
                if entry.get('thread'):
                    outputs = p.run_threaded(*inputs)
                else:
                    outputs = p.run(*inputs)

                # save the output to memory
                if outputs is not None:
                    self.telemetry.put(entry['outputs'], outputs)
                # finish timing part run
                self.profiler.on_part_finished(p)

    def stop(self):        
        print('Shutting down vehicle and its parts...')
        for entry in self.parts:
            print(entry['part_name'])
            try:
                entry['part'].shutdown()
                if entry.get('threaded'):
                    print("deleting thread")
                    del entry['thread']
                    print("thread deleted")
            except AttributeError:
                # usually from missing shutdown method, which should be optional
                pass
            except Exception as e:
                print(e)

        self.telemetry.cleanup_postsession()
        self.profiler.report()
