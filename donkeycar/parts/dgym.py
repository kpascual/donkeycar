import os
import os
import time
import gym
import gym_donkeycar

def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

class DonkeyGymEnv(object):

    def __init__(self, sim_path, host="127.0.0.1", port=9091, headless=0, env_name="donkey-generated-track-v0", sync="asynchronous", conf={}, delay=0):
        os.environ['DONKEY_SIM_PATH'] = sim_path
        os.environ['DONKEY_SIM_PORT'] = str(port)
        os.environ['DONKEY_SIM_HEADLESS'] = str(headless)
        os.environ['DONKEY_SIM_SYNC'] = str(sync)

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
        self.info = { 'pos' : (0., 0., 0.), 'speed': 0.0}
        self.delay = float(delay)


    def prestart(self):
        self.running = True

        self.env = gym.make(self.env_name, exe_path=self.sim_path, host=self.host, port=self.port)
        self.frame = self.env.reset()

        if "body_style" in self.conf:
            self.env.viewer.set_car_config(self.conf["body_style"], self.conf["body_rgb"], self.conf["car_name"], self.conf["font_size"])
            #without this small delay, we seem to miss packets
            time.sleep(0.1)

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

        return self.frame, self.info['speed'], self.info['pos']

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.env.close()


    
