#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 10:44:24 2017

@author: wroscoe
"""

import time
import os
from itertools import cycle
from statistics import median
from threading import Thread
import threading
from .memory import Memory
from prettytable import PrettyTable
from donkeycar.parts.datastore import TubHandler

# How quickly do certain parts run? (e.g. is the drive loop efficient?)
# This is different than data outputted by the parts
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


class DataRecorder:
    def __init__(self, root_path):
        self.root_path = root_path
        self.mem = Memory()
        self.path = None

        self.inputs = ()
        self.types = ()
        self.tub = None
        self.tub_handler = TubHandler(path=self.root_path)


    def create_tub(self):
        print(self.inputs)
        print(self.types)
        self.tub = self.tub_handler.new_tub_writer(inputs=self.inputs, types=self.types)

    def set_path(self, path):
        self.path = path
        self.tub_handler.path = os.path.join(self.root_path, self.path)

    def initialize_recorder(self):
        # Create tub
        self.create_tub()
        # Initialize default channel values
        for channel_name, data_type in zip(self.inputs, self.types):
            if data_type == 'float':
                self.put([channel_name], 0.0)
                print("channel name initialized")
                print(channel_name)
            


    def save_vehicle_configuration(self, parts):
        newparts = []
        for p in parts:
            newparts.append({k:v for k,v in p.items() if k != 'thread' and k != 'part'})

        self.tub.parts = newparts

    def save_end_time(self):
        self.tub.end_time = time.time()

    def set_inputs(self, inputs):
        print("set inputs")
        print(inputs)
        self.inputs = inputs

    def set_types(self, types):
        print("set types")
        print(types)
        self.types = types

    def record(self):
        inputs = self.mem.get(self.inputs)
        self.tub.run(*inputs)

    def get(self, keys):
        return self.mem.get(keys)

    def put(self, keys, inputs):
        self.mem.put(keys, inputs)

    def cleanup_postsession(self):
        if self.tub is not None:
            self.save_end_time()
            self.tub.write_meta()


class Vehicle:
    def __init__(self, data_path, mem=None):

        if not mem:
            mem = Memory()
        self.mem = mem
        self.parts = []
        self.driver = None
        self.on = True
        self.threads = []
        self.profiler = PartProfiler()
        self.data_recorder = DataRecorder(data_path)
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

    def set_channels(self, channels):
        self.channels = channels.copy()
        print(self.channels)

    def set_data_recorder_config(self, inputs=None, path=None):
        print("set data recorder config")
        if inputs is not None:
            print("inputs:")
            print(inputs)
            print("existing channels:")
            print(self.channels)
            inputs = [c[0] for c in self.channels if c[0] in inputs]
            types = [c[1] for c in self.channels if c[0] in inputs]
            print(inputs)
            print(types)
            self.data_recorder.set_inputs(inputs)
            self.data_recorder.set_types(types)

        if path is not None:
            self.data_recorder.set_path(path)


    def add_part(self, part, inputs=[], outputs=[], threaded=False, run_condition=None):
        self.add(part, inputs, outputs, threaded=threaded, run_condition=run_condition)

    def initialize_channel_data(self, channels, values):
        for channel_name, data_type in self.channels:
            if data_type == 'float':
                self.data_recorder.put([channel_name], [0.0])

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
                if hasattr(entry['part'], 'prestart'):
                    entry['part'].prestart()

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
            self.data_recorder.initialize_recorder()
            self.data_recorder.save_vehicle_configuration(self.parts)

            # wait until the parts warm up.
            print('Starting vehicle...')

            loop_count = 0
            while self.on and getattr(ct, "is_on", True):
                start_time = time.time()
                loop_count += 1

                # update all third party sensors
                self.update_parts()
                # update system parts

                # record telemetry
                if self.data_recorder.get(['recording'])[0]:
                    self.data_recorder.record()

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
                run = self.data_recorder.get([run_condition])[0]
            
            if run:
                # get part
                p = entry['part']
                # start timing part run
                self.profiler.on_part_start(p)
                # get inputs from memory
                inputs = self.data_recorder.get(entry['inputs'])
                # run the part
                if entry.get('thread'):
                    outputs = p.run_threaded(*inputs)
                else:
                    outputs = p.run(*inputs)

                # save the output to memory
                if outputs is not None:
                    self.data_recorder.put(entry['outputs'], outputs)
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

        self.data_recorder.cleanup_postsession()
        self.profiler.report()
