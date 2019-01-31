import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from gan import Gan

from random import randint
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
import sys

import cv2
import numpy as np
from PIL import Image

def changeColorSpace(image):
    image = np.array(image)
    hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    return Image.fromarray(hsv_image)

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

        dataGen = ImageDataGenerator(
            #preprocessing_function=changeColorSpace
        )
        self.posters = dataGen.flow_from_directory(
            self.dataFolder,
            target_size = (self.img_rows, self.img_cols),
            batch_size = 16,
            class_mode = 'categorical',
            color_mode = 'grayscale' if grayscale else 'rgb'
        )
        return self.posters
    def generate_noise(self, n_samples, noise_dim):
        X = np.random.normal(0, 1, size=(n_samples, noise_dim))
        return X

    def show_imgs(self,batchidx):
        noise = self.generate_noise(9, 100)
        gen_imgs = self.generator.predict(noise)

        fig, axs = plt.subplots(3, 3)
        count = 0
        for i in range(3):
            for j in range(3):
            # Dont scale the images back, let keras handle it
                img = image.array_to_img(gen_imgs[count], scale=True)
                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                count += 1
        plt.show()

    def train(self, N_EPOCHS=100, batch_size=16, save_interval = 200, NB_DATA = 3866):
        

        N_EPOCHS = 100
        num_batches = int(NB_DATA/batch_size)
        for epoch in range(N_EPOCHS):
            print("Epoch ", epoch)
            cum_d_loss = 0.
            cum_g_loss = 0.
            
            for batch_idx in tqdm(range(num_batches), file=sys.stdout):
                # Get the next set of real images to be used in this iteration
                #images = X_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
                (images, labels_real) = self.posters.next()
                if(images.shape[0] != batch_size):
                    print('reset')
                    self.posters.reset()
                    (images, labels_real) = self.posters.next()
                noise_data = self.generate_noise(batch_size, 100)
                generated_images = self.generator.predict(noise_data)

                # Train on soft labels (add noise to labels as well)
                noise_prop = 0.05 # Randomly flip 5% of labels
                
                # Prepare labels for real data
                true_labels = np.zeros((batch_size, 1)) + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
                flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
                true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
                
                # Train discriminator on real data
                d_loss_true = self.discriminator.train_on_batch(images, true_labels)

                # Prepare labels for generated data
                gene_labels = np.ones((batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
                flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
                gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
                
                # Train discriminator on generated data
                d_loss_gene = self.discriminator.train_on_batch(generated_images, gene_labels)

                # d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
                # cum_d_loss += d_loss

                # Train generator
                noise_data = self.generate_noise(batch_size, 100)
                g_loss = self.adversarial.train_on_batch(noise_data, np.zeros((batch_size, 1)))
                # cum_g_loss += g_loss

            print('  Epoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch+1, 0, 0))
            self.show_imgs("epoch" + str(epoch))
        # for i in range(train_steps):
        #     noise_input = np.random.uniform(0, 1.0, size=[batch_size,100]) #Fake image will be generated from a noise TODO : make sure the noise shape is OK
        #     # TODO : Generate fake images using the noise and pick true images from the dataset
        #     # train the discriminator on the true and fake images (2 * batch_size) label 0/1

        #     (images_real, labels_real) = self.posters.next()
        #     images_fake = self.gan.generator().predict(noise_input)

        #     x = np.concatenate((images_real ,images_fake))
        #     y = np.zeros([2 * batch_size, 1]) + np.random.uniform(low=0.0, high=0.1, size=(2 * batch_size, 1))
        #     y[batch_size:,:] = 1 + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))

        #     print(x.shape)
        #     if(x.shape[0] != 2*batch_size):
        #         self.posters.reset()
        #         (images_real, labels_real) = self.posters.next()
        #         images_fake = self.gan.generator().predict(noise_input)

        #         x = np.concatenate((images_real ,images_fake))
        #         y = np.zeros([2 * batch_size, 1]) + np.random.uniform(low=0.0, high=0.1, size=(2 * batch_size, 1))
        #         y[batch_size:,:] = 1 + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
            
        #     noise_prop = 0.05 # Randomly flip 5% of labels
        #     flipped_idx = np.random.choice(np.arange(len(y)), size=int(noise_prop*len(y)))
        #     y[flipped_idx] = 1 - y[flipped_idx]


        #     print(y.shape)

        #     d_loss = self.discriminator.train_on_batch(x,y)
        #     #TODO : Generate a new noise and train the adversarial model with label 1
        #     noise_input = np.random.uniform(-1.0, 1.0, size=[batch_size,100]) #Fake image will be generated from a noise TODO : make sure the noise shape is OK
        #     y = np.ones([batch_size, 1])
        #     if(d_loss[1] > 0.65):
        #         a_loss = self.adversarial.train_on_batch(noise_input, y)

        #     log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        #     if(d_loss[1] > 0.65):
        #         log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        #     print(log_mesg)
        #     if (i+1)%save_interval==0:
        #             self.plot_images(save2file=True, samples=noise_input.shape[0],\
        #                 noise=noise_input, step=(i+1))
    
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
            plt.subplot(4, 4, i+1)
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
