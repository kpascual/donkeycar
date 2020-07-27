#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Uses the data written by the donkey v2.2 tub writer,
but faster training with proper sampling of distribution over tubs. 


Usage:
    train.py --driver=<driver_name> --nn=<nn> --data=<tub1,tub2,..tubn>  [--config=<config>]

Options:
    -h --help              Show this screen.
"""
import os
import glob
import shutil
import random
import json
import time
import zlib
from os.path import basename, join, splitext, dirname
import pickle
import datetime
import importlib

from tensorflow.python import keras
from docopt import docopt
import numpy as np
from PIL import Image
import yaml

import donkeycar as dk
from donkeycar.parts.datastore import Tub
from donkeycar.parts.keras import KerasLinear, KerasIMU,\
     KerasCategorical, KerasBehavioral, Keras3D_CNN,\
     KerasRNN_LSTM, KerasLatent, KerasLocalizer
from donkeycar.parts.augment import augment_image
from donkeycar.utils import *
from donkeycar import utils
from nn import linear
import matplotlib.pyplot as plt

figure_format = 'png'



'''
Tub management
'''
def split_test_train(record_paths, train_test_split):
    paths = record_paths.copy()

    #  Ratio of samples to use as training data, the remaining are used for evaluation
    random.shuffle(paths)
    target_train_count = int(train_test_split * len(paths))
    train = paths[:target_train_count].copy()
    test = paths[target_train_count:].copy()

    return train, test


def collate_records_updated(records):
    new_records = []

    # check whether data lives in associated file, or lives in record path
    for tub, path in records:
        #basepath = os.path.dirname(path)        

        try:
            with open(path, 'r') as fp:
                json_data = json.load(fp)
        except:
            continue
        newsample = tub.read_record(json_data)

        new_records.append(newsample)

    return new_records
        

class MyCPCallback(keras.callbacks.ModelCheckpoint):
    '''
    custom callback to interact with best val loss during continuous training
    '''

    def __init__(self, send_model_cb=None, cfg=None, *args, **kwargs):
        super(MyCPCallback, self).__init__(*args, **kwargs)
        self.reset_best_end_of_epoch = False
        self.send_model_cb = send_model_cb
        self.last_modified_time = None
        self.cfg = cfg

    def reset_best(self):
        self.reset_best_end_of_epoch = True

    def on_epoch_end(self, epoch, logs=None):
        super(MyCPCallback, self).on_epoch_end(epoch, logs)

        if self.send_model_cb:
            '''
            check whether the file changed and send to the pi
            '''
            filepath = self.filepath.format(epoch=epoch, **logs)
            if os.path.exists(filepath):
                last_modified_time = os.path.getmtime(filepath)
                if self.last_modified_time is None or self.last_modified_time < last_modified_time:
                    self.last_modified_time = last_modified_time
                    self.send_model_cb(self.cfg, self.model, filepath)

        '''
        when reset best is set, we want to make sure to run an entire epoch
        before setting our new best on the new total records
        '''        
        if self.reset_best_end_of_epoch:
            self.reset_best_end_of_epoch = False
            self.best = np.Inf
        

def on_best_model(cfg, model, model_filename):
    model.save(model_filename, include_optimizer=False)
        
    
def generator(records, batch_size, preprocess_X, preprocess_y):
    while True:

        random.shuffle(records)
        batch_data = []
        for tub, path in records:

            batch_data.append((tub, path))

            if len(batch_data) == batch_size:
                dataset = collate_records_updated(batch_data)

                y = preprocess_y(dataset)
                X = preprocess_X(dataset)

                yield X, y

                batch_data = []
    

def train(cfg, tub_names, driver_name, nn):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    ''' 
    verbose = cfg['VERBOSE_TRAIN']

    # Create driver directory
    driver_path = os.path.join(cfg['DRIVERS_PATH'], driver_name)
    if not os.path.isdir(driver_path):
        os.mkdir(driver_path)
        open(driver_path + '/__init__.py','w').close() # create empty init so can be called later in drive mode

    # Save training parameters
    # 1) tubs 2) keras model 3) hyperparameters
    training = {
        'keras_model': nn,
        'data': tub_names
    }
    f = open(driver_path + '/training.json', 'w')
    f.write(json.dumps(training))
    f.close()

    # Copy keras model to driver folder
    source_keras_model_path = os.path.join(cfg['KERAS_MODEL_PATH'], nn + '.py')
    dest_keras_model_path = os.path.join(driver_path, nn + '.py')
    shutil.copyfile(source_keras_model_path, dest_keras_model_path)
    
    model_name = driver_name + '/model.h5'
    
    keras_model = importlib.import_module('nn.' + nn)
    kl = keras_model.get()
    print('training with model type', type(kl))

    if cfg['OPTIMIZER']:
        kl.set_optimizer(cfg['OPTIMIZER'], cfg['LEARNING_RATE'], cfg['LEARNING_RATE_DECAY'])
    kl.compile()

    if cfg['PRINT_MODEL_SUMMARY']:
        print(kl.model.summary())
    
    extract_data_from_pickles(cfg['DATA_PATH'], tub_names)

    # Get a list of all tubs
    tub_paths = utils.gather_tub_paths(cfg['DATA_PATH'], tub_names)
    tubs = tuple([Tub(p) for p in tub_paths])

    records = []
    for tub in tubs:
        record_paths = tub.gather_records()
        records += [(tub, path) for path in record_paths]

    #records = utils.gather_records(, tub_names, verbose=True)
    #print('collating %d records ...' % (len(records)))
    train_indexes, test_indexes = split_test_train(records, cfg['TRAIN_TEST_SPLIT'])

    model_path = os.path.expanduser(model_name)
    model_path = os.path.join(cfg['DRIVERS_PATH'], model_path)
    
    #checkpoint to save model after each epoch and send best to the pi.
    save_best = MyCPCallback(
        send_model_cb = on_best_model,
        filepath = model_path,
        monitor = 'val_loss', 
        verbose = verbose, 
        save_best_only = True, 
        mode = 'min',
        cfg = cfg
       )

    train_gen = generator(train_indexes, cfg['BATCH_SIZE'], kl.preprocess_X, kl.preprocess_y)
    val_gen = generator(test_indexes, cfg['BATCH_SIZE'], kl.preprocess_X, kl.preprocess_y)
    


    print("train: %d, val: %d" % (len(train_indexes), len(test_indexes)))
    print('total records: %d' %(len(train_indexes) + len(test_indexes)))
    
    steps_per_epoch = len(train_indexes) // cfg['BATCH_SIZE']
    
    val_steps = len(test_indexes) // cfg['BATCH_SIZE']
    print('steps_per_epoch', steps_per_epoch)



    go_train(kl, cfg, train_gen, val_gen, model_name, steps_per_epoch, val_steps, verbose, save_best)

    
    
def go_train(kl, cfg, train_gen, val_gen, model_name, steps_per_epoch, val_steps, verbose, save_best=None):

    start = time.time()

    model_path = os.path.expanduser(model_name)
    model_path = os.path.join(cfg['DRIVERS_PATH'], model_path)

    #checkpoint to save model after each epoch and send best to the pi.
    if save_best is None:
        save_best = MyCPCallback(send_model_cb=on_best_model,
                                    filepath=model_path,
                                    monitor='val_loss', 
                                    verbose=verbose, 
                                    save_best_only=True, 
                                    mode='min',
                                    cfg=cfg)

    #stop training if the validation error stops improving.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                min_delta=cfg['MIN_DELTA'], 
                                                patience=cfg['EARLY_STOP_PATIENCE'], 
                                                verbose=verbose, 
                                                mode='auto')

    if steps_per_epoch < 2:
        raise Exception("Too little data to train. Please record more records.")

    epochs = cfg['MAX_EPOCHS']

    workers_count = 1
    use_multiprocessing = False

    callbacks_list = [save_best]

    if cfg['USE_EARLY_STOP']:
        callbacks_list.append(early_stop)

    history = kl.model.fit_generator(
        train_gen, 
        steps_per_epoch = steps_per_epoch, 
        epochs = epochs, 
        verbose = cfg['VERBOSE_TRAIN'], 
        validation_data = val_gen,
        callbacks = callbacks_list, 
        validation_steps = val_steps,
        workers = workers_count,
        use_multiprocessing = use_multiprocessing
    )
                    
    full_model_val_loss = min(history.history['val_loss'])
    max_val_loss = full_model_val_loss + cfg['PRUNE_VAL_LOSS_DEGRADATION_LIMIT']

    duration_train = time.time() - start
    print("Training completed in %s." % str(datetime.timedelta(seconds=round(duration_train))) )

    print("\n\n----------- Best Eval Loss :%f ---------" % save_best.best)

    if cfg['SHOW_PLOT']:
        try:
            plt.figure(1)

            # Only do accuracy if we have that data (e.g. categorical outputs)
            if 'angle_out_acc' in history.history:
                plt.subplot(121)

            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validate'], loc='upper right')
            
            # summarize history for acc
            if 'angle_out_acc' in history.history:
                plt.subplot(122)
                plt.plot(history.history['angle_out_acc'])
                plt.plot(history.history['val_angle_out_acc'])
                plt.title('model angle accuracy')
                plt.ylabel('acc')
                plt.xlabel('epoch')
                #plt.legend(['train', 'validate'], loc='upper left')

            plt.savefig(model_path + '_loss_acc_%f.%s' % (save_best.best, figure_format))
            plt.show()
        except Exception as ex:
            print("problems with loss graph: {}".format( ex ) )


def extract_data_from_pickles(root_path, tubs):
    """
    Extracts record_{id}.json and image from a pickle with the same id if exists in the tub.
    Then writes extracted json/jpg along side the source pickle that tub.
    This assumes the format {id}.pickle in the tub directory.
    :param cfg: config with data location configuration. Generally the global config object.
    :param tubs: The list of tubs involved in training.
    :return: implicit None.
    """
    t_paths = utils.gather_tub_paths(root_path, tubs)
    for tub_path in t_paths:
        file_paths = glob.glob(join(tub_path, '*.pickle'))
        print('found {} pickles writing json records and images in tub {}'.format(len(file_paths), tub_path))
        for file_path in file_paths:
            # print('loading data from {}'.format(file_paths))
            with open(file_path, 'rb') as f:
                p = zlib.decompress(f.read())
            data = pickle.loads(p)
           
            base_path = dirname(file_path)
            filename = splitext(basename(file_path))[0]
            image_path = join(base_path, filename + '.jpg')
            img = Image.fromarray(np.uint8(data['val']['cam/image_array']))
            img.save(image_path)
            
            data['val']['cam/image_array'] = filename + '.jpg'

            with open(join(base_path, 'record_{}.json'.format(filename)), 'w') as f:
                json.dump(data['val'], f)


if __name__ == "__main__":
    args = docopt(__doc__)
    cfg = dk.load_config()

    driver = args['--driver']
    tubs = args['--data']
    nn = args['--nn']
    config = args['--config']

    if config:
        newcfg = yaml.load(open(config,'r'))
    else:
        newcfg = yaml.load(open('training_defaults.yml','r'))
        print(newcfg)

    dirs = []
    if tubs is not None:
        tub_paths = [os.path.expanduser(n) for n in tubs.split(',')]
        dirs.extend( tub_paths )

    train(newcfg, dirs, driver, nn)
