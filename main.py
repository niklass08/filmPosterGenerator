import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator
from random import randint

import matplotlib.pyplot as plt


class Gan():
    def __init__(self, img_rows = 28, img_cols = 28, channels = 1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
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
        self.dataFolder = dataFolder
        # TODO : create a data generator and read files from dataFolder in order to generate train set and validation set
        # maybe use one hot encoding for the class
        self.posters = None

        self.gan = Gan()
    def load_data(self, grayscale = True):

        dataGen = ImageDataGenerator()
        data_set = dataGen.flow_from_directory(
            self.dataFolder,
            target_size = (self.img_rows, self.img_cols),
            batch_size = 32,
            class_mode = 'categorical',
            color_mode = 'grayscale' if grayscale else 'rgb'
        )
        return data_set

    def train(self, train_steps=500, batch_size=256):
        
        for i in range(train_steps):
            noise_input = np.random.uniform(-1.0, 1.0, size=[batch_size,100]) #Fake image will be generated from a noise TODO : make sure the noise shape is OK
            # TODO : Generate fake images using the noise and pick true images from the dataset
            # train the discriminator on the true and fake images (2 * batch_size) label 0/1


            #TODO : Generate a new noise and train the adversarial model with label 1

grayscale = True
fpg = filmPosterGan(rows = 64, cols = 64, channels = 1, dataFolder = "./data")
data_gen = fpg.load_data()
print(data_gen)
(x, y) = data_gen.next()
print(x.shape)

i = randint(1,31)
print(x[i].shape)
image = x[i]
if grayscale: image = np.reshape(image, (64, 64))
print("Dimension of input image=",np.shape(image))

#plt.imshow(image.transpose(2,1,0))
if i>0: plt.figure()
if grayscale:
    plt.imshow(image, cmap='gray')
else:
    plt.imshow(image)
plt.show()