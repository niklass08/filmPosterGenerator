import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from gan import Gan

from random import randint
import matplotlib.pyplot as plt
class filmPosterGan():
    def __init__(self, rows, cols, channels, dataFolder):
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.dataFolder = dataFolder
        # TODO : create a data generator and read files from dataFolder in order to generate train set and validation set
        # maybe use one hot encoding for the class
        self.posters = None

        self.gan = Gan(rows, cols, channels)
        self.discriminator = self.gan.discriminator_model()
        self.adversarial = self.gan.adversarial_model()
        self.generator = self.gan.generator()
    def load_data(self, grayscale = True):

        dataGen = ImageDataGenerator()
        self.posters = dataGen.flow_from_directory(
            self.dataFolder,
            target_size = (self.img_rows, self.img_cols),
            batch_size = 32,
            class_mode = 'categorical',
            color_mode = 'grayscale' if grayscale else 'rgb'
        )
        return self.posters

    def train(self, train_steps=500, batch_size=32, save_interval = 50):
        
        for i in range(train_steps):
            noise_input = np.random.uniform(-1.0, 1.0, size=[batch_size,100]) #Fake image will be generated from a noise TODO : make sure the noise shape is OK
            # TODO : Generate fake images using the noise and pick true images from the dataset
            # train the discriminator on the true and fake images (2 * batch_size) label 0/1

            (images_real, labels_real) = self.posters.next()
            images_fake = self.gan.generator().predict(noise_input)

            x = np.concatenate((images_real ,images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:,:] = 0

            print(x.shape)
            if(x.shape[0] != 2*batch_size):
                self.posters.reset()
                (images_real, labels_real) = self.posters.next()
                images_fake = self.gan.generator().predict(noise_input)

                x = np.concatenate((images_real ,images_fake))
                y = np.ones([2 * batch_size, 1])
                y[batch_size:,:] = 0
            print(y.shape)

            d_loss = self.discriminator.train_on_batch(x,y)
            #TODO : Generate a new noise and train the adversarial model with label 1
            noise_input = np.random.uniform(-1.0, 1.0, size=[batch_size,100]) #Fake image will be generated from a noise TODO : make sure the noise shape is OK
            y = np.ones([batch_size, 1])

            a_loss = self.adversarial.train_on_batch(noise_input, y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))
    
    def plot_images(self, save2file=False, samples=16, noise=None, step=0):
        filename = 'posters.png'
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "posters_%d.png" % step
        images = self.generator.predict(noise)

        plt.figure(figsize=(10,10))
        print("images[0] shape")
        print(images.shape[0])
        for i in range(images.shape[0]):
            plt.subplot(4, 8, i+1)
            image = images[i, :, :, :]
            if(self.channels == 1):
                image = np.reshape(image, [self.img_rows, self.img_cols])
                plt.imshow(image, cmap='gray')
            else:
                image =255 * np.reshape(image, [self.img_rows, self.img_cols, self.channels])
                plt.imshow(image.astype('uint8'))
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()