import os
import os
import time
import gym
import gym_donkeycar

def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

class DonkeyGymEnv(object):

    def __init__(self, sim_path, host="127.0.0.1", port=9091, headless=0, env_name="donkey-generated-track-v0", sync="asynchronous", conf={}, delay=0):

        if sim_path != "remote":
            if not os.path.exists(sim_path):
                raise Exception("The path you provided for the sim does not exist.") 

            if not is_exe(sim_path):
                raise Exception("The path you provided is not an executable.") 

        self.host = host
        self.port = port
        self.sim_path = sim_path
        self.env_name = env_name
        self.conf = conf
        self.action = [0.0, 0.0]
        self.running = False
        self.info = { 
            'pos' : (0., 0., 0.), 
            'gyro' : (0., 0., 0.), 
            'accel' : (0., 0., 0.), 
            'vel' : (0., 0., 0.), 
            'speed': 0.0
        }
        self.delay = float(delay)


    def prestart(self):
        self.running = True
        conf = {
            "exe_path": self.sim_path,
            "host": self.host,
            "port": self.port
        }
        self.conf['exe_path'] = self.sim_path
        self.conf['host'] = self.host
        self.conf['port'] = self.port

        self.env = gym.make(self.env_name, conf=self.conf)
        self.frame = self.env.reset()


    def update(self):
        while self.running:
            self.frame, _, _, self.info = self.env.step(self.action)

    def run_threaded(self, steering, throttle):
        if steering is None or throttle is None:
            steering = 0.0
            throttle = 0.0
        if self.delay > 0.0:
            time.sleep(self.delay / 1000.0)
        self.action = [steering, throttle]

        return self.frame, self.info['speed'], self.info['pos'], self.info['gyro'], self.info['accel'], self.info['vel']

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.env.close()


    
