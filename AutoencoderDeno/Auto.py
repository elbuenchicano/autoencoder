import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from utils_video_image import plot_chart, psnr

#Callback to print learning rate decaying over the epoches
class LearningRatePrinter(Callback):
    def init(self):
        super(LearningRatePrinter, self).init()

    def on_epoch_begin(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print ("learning rate -> " + str(lr))

################################################################################
################################################################################
#Main class autencoder
class Autoencoder(object):
    def __init__(this, input_shape):
        this.createNetwork2D(input_shape)

    def createNetwork2D(this, input_shape):
        this.model = Sequential()
        #Input
        #input_img = shape=(28, 28, 1)
        this.model.add(ZeroPadding2D((0,0),input_shape=input_shape))
        
        #Convolution and Maxpooling
        this.model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
        this.model.add(MaxPooling2D((2,2)))
        this.model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
        this.model.add(MaxPooling2D((2,2)))
        #this.model.add(Convolution2D(8, 3, 3, activation='relu', border_mode='same'))

        #Deconvolution and Upsampling
        this.model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
        this.model.add(UpSampling2D((2,2)))
        this.model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
        this.model.add(UpSampling2D((2,2)))
        #this.model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
        
        #Output
        this.model.add(Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same'))
        this.model.summary()

    ############################################################################
    def evaluate(this, x_test, x_test_original, model_in, n, shape):
        this.model.load_weights(model_in)
        a = this.model.predict(x_test)

        lst = []
        titles = []

        for i in range(n):
            original = x_test_original[i].reshape(shape) 
            reconst = a[i].reshape(shape) 
            noisy = x_test[i].reshape(shape) 

            lst.append(original)
            lst.append(reconst)
            lst.append(noisy)
            
            titles.append( "%.2f" % psnr(original, noisy))
            titles.append( '' )
            titles.append( '' )

        plot_chart(lst, n, 3, ax_vis = False, gray = True, titles=titles, title = 'Psnr')
        
    ############################################################################
    def train(this, X_train, X_train_, X_test, X_test_, batch_size, epochs, out):
        
        #Stochastic Gradient Descent to perform backpropagation 
        #(decay = lr decay rate)
        sgd = SGD(lr=0.001, momentum=0.0, decay=0.00001, nesterov=False)
        #Save weights
        weights_save = ModelCheckpoint(filepath='weights.h5', 
                                       monitor='val_loss', 
                                       verbose=0, 
                                       save_best_only=True, 
                                       save_weights_only=False, 
                                       mode='min')
        
        lr_printer = LearningRatePrinter()

        #run autoencoder
        this.model.compile(loss='binary_crossentropy', optimizer='adadelta')

        this.model.fit(X_train, X_train_, 
                  batch_size=batch_size, 
                  nb_epoch=epochs,
                  verbose=1, 
                  shuffle=True,
                  validation_data=(X_test, X_test_), 
                  callbacks=[lr_printer, weights_save])

        this.model.evaluate(X_test, X_test_, verbose=0)

#END CLASS
################################################################################
################################################################################


