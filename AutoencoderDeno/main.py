
import numpy as np
import argparse
import json
import cv2
import matplotlib.pyplot as plt

from keras.datasets import mnist
from utils import u_getPath, u_listFileAll
from utils_video_image import plot_chart
from Auto import Autoencoder
from sklearn.model_selection import train_test_split

################################################################################
################################################################################
def getNoisyDataMnist(noise_factor):
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    shape = (28, 28, 1)
    return [x_train, x_test, x_train_noisy, x_test_noisy, shape];

################################################################################
################################################################################
def getNoisyData(img_path, token, noise, auto_type):
    
    files = u_listFileAll(img_path, token)
    clean = []
    
    flag = 1 if auto_type  else 0
    for file in files:
        print('Reading file: ', file)
        img         = cv2.imread(file, flag)/255
        clean.append(img)
    
    #get dimensions
    shape = clean[0].shape
    if not auto_type: shape += (1,)

    #reshaping to net
    clean   = np.reshape(clean, (len(clean), shape[0], shape[1], shape[2])) 

    #adding noise to clean images
    noisy   = clean + noise * np.random.normal(loc=0.0, scale=1.0, size=clean.shape) 
    
    #clipping 0 1 
    noisy   = np.clip(noisy, 0., 1.)

    #dividing in test and train
    x_train, x_test, y_train, y_test = train_test_split(
        clean, noisy, test_size=0.3, random_state=0)

    return x_train, x_test, y_train, y_test, shape;

################################################################################
################################################################################
def train(general, data):
  
    #noise factor that will be added to images
    noise       = general['noise']

    model_out   = general['model_name']

    #this flag is to define the autoencoder type 0 = 2D, 1 = 3D
    auto_type   = general['auto_type']

    bach_size   = data['batch_size']
    epochs      = data['epochs']
    img_path    = data['img_path']
    img_token   = data['img_token']

    #x_train, x_test, x_train_noisy, x_test_noisy, shape = getNoisyDataMnist(noise)
    x_train, x_test, x_train_noisy, x_test_noisy, shape = getNoisyData(
        img_path, img_token, noise, auto_type)

    #plot_chart([x_train[0].reshape(128,128), x_train_noisy[0].reshape(128,128)], 1, 2)

    autoencoder = Autoencoder(shape)
    autoencoder.train(X_train   = x_train_noisy, 
                      X_train_  = x_train,
                      epochs    = epochs,
                      batch_size= bach_size,
                      X_test    = x_test_noisy, 
                      X_test_   = x_test,
                      out       = model_out)
    
    print('Saving model in: ', model_out)

################################################################################
################################################################################
def test(general, data):

    #noise factor that will be added to images
    noise       = general['noise']

    model_in    = general['model_name']

    #this flag is to define the autoencoder type 0 = 2D, 1 = 3D
    auto_type   = general['auto_type']

    ntest       = data['ntest']
    img_path    = data['img_path']
    img_token   = data['img_token']

    #x_train, x_test, x_train_noisy, x_test_noisy, shape = getNoisyDataMnist(noise)
    x_train, x_test, x_train_noisy, x_test_noisy, shape = getNoisyData(
        img_path, img_token, noise, auto_type)

    autoencoder = Autoencoder(shape)

    if shape[2] == 1:
        shape = (shape[0], shape[1])

    autoencoder.evaluate(x_test_noisy, x_test, model_in, ntest, shape)


################################################################################
################################################################################
################################ Main controler ################################
def _main():

    funcdict = {'train' : train,
                'test'  : test}

    conf_f  = u_getPath()
    confs   = json.load(open(conf_f))

    #...........................................................................
    funcdict[confs['op_type']](confs['general'], confs[confs['op_type']])
    
   
################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    _main()