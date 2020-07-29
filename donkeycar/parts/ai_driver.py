import importlib
from tensorflow.python import keras


class AiDriver:

    def __init__(self, driver_root_path, driver_name):
        self.driver = driver_name
        self.root_path = driver_root_path
        self.klib_part = importlib.import_module(self.root_path + '.' + driver_name + '.part')

    def prestart(self):
        self.model = keras.models.load_model(self.root_path + '/' + self.driver + '/model.h5')

    def run(self, *args):
        angle, throttle = self.klib_part.run(self.model, *args)

        return float(angle), float(throttle)