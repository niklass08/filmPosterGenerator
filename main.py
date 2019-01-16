import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop


class Gan():
    def __init__(self, img_rows = 28, img_cols = 28, channels = 1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
    
    def discriminator(self):
        # TODO : add discriminator topology
        return self.D
    
    def generator(self):
        # TODO : add generator topology
        return self.G
    
    def discriminator_model(self):
        # TODO : Compile the discriminator : choose Optimizer and loss function
        return self.DM

    def adversarial_model(self):
        #TODO : Add generator + discriminator + choose optimizer and losse function
        return self.AM

class filmPosterGan():
    def __init__(self, rows, cols, channels, dataFolder):
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        # TODO : create a data generator and read files from dataFolder in order to generate train set and validation set
        # maybe use one hot encoding for the class
        self.posters = None

        self.gan = Gan()

    def train(self, train_steps=500, batch_size=256)
        
        for i in range(train_steps):
            noise_input = np.random.uniform(-1.0, 1.0, size=[batch_size,100]) #Fake image will be generated from a noise TODO : make sure the noise shape is OK
            # TODO : Generate fake images using the noise and pick true images from the dataset
            # train the discriminator on the true and fake images (2 * batch_size) label 0/1


            #TODO : Generate a new noise and train the adversarial model with label 1
