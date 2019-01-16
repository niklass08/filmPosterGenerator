import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
from keras.preprocessing.image import ImageDataGenerator
from random import randint
import matplotlib.pyplot as plt
from gan import Gan

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

g = Gan()
g.discriminator()
