import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from gan import Gan

class filmPosterGan():
    def __init__(self, rows, cols, channels, dataFolder):
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        # TODO : create a data generator and read files from dataFolder in order to generate train set and validation set
        # maybe use one hot encoding for the class
        self.posters = None

        self.gan = Gan()

    def train(self, train_steps=500, batch_size=256):
        for i in range(train_steps):
            noise_input = np.random.uniform(-1.0, 1.0, size=[batch_size,100]) #Fake image will be generated from a noise TODO : make sure the noise shape is OK
            # TODO : Generate fake images using the noise and pick true images from the dataset
            # train the discriminator on the true and fake images (2 * batch_size) label 0/1


            #TODO : Generate a new noise and train the adversarial model with label 1

g = Gan(64,64,3)
g.discriminator()
g.generator()
