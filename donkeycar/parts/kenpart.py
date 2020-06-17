import random
from bluepy.btle import Scanner

class Beacons:
    def __init__(self):
        self.beacon1 = 0.0
        self.beacon2 = 0.0
        self.beacon3 = 0.0
        self.scanner = Scanner()


    #1 - 18:04:ed:51:8e:dd
    #2 - 18:04:ed:51:70:62
    #3 - 18:04:ed:51:72:07
    def run(self):
        devices = self.scanner.scan(1.0)
        beacon1 = -100
        beacon2 = -100
        beacon3 = -100
        for d in devices:
            if d.addr == '18:04:ed:51:8e:dd':
                beacon1 = d.rssi
            elif d.addr == '18:04:ed:51:70:62':
                beacon2 = d.rssi
            elif d.addr == '18:04:ed:51:72:07':
                beacon3 = d.rssi

        randomval = random.random()
            
        return beacon1, beacon2, beacon3


    def update(self):
        while True:
            self.beacon1, self.beacon2, self.beacon3 = self.run()


    def run_threaded(self):
        return self.beacon1, self.beacon2, self.beacon3
