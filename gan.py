from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

class Gan():
    def __init__(self, img_rows = 28, img_cols = 28, channels = 1, filters = 32):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.filters = filters
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        # TODO : add discriminator topology
        if self.D:
            return self.D

        self.D = Sequential()
        dropout = 0.4
        filters = 128
        input_shape = (self.img_rows, self.img_cols, self.channels)
        self.D.add(Conv2D(filters, 5, padding='same', data_format='channels_last', input_shape=input_shape, strides=1))
        self.D.add(LeakyReLU(alpha=0.1))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(filters, 5, padding='same', strides=2))
        self.D.add(LeakyReLU(alpha=0.1))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(filters, 5, padding='same', strides=2))
        self.D.add(LeakyReLU(alpha=0.1))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(filters, 5, padding='same', strides=2))
        self.D.add(LeakyReLU(alpha=0.1))
        self.D.add(Dropout(dropout))

        self.D.add(Flatten())
        self.D.add(Dense(1, activation='sigmoid'))

        self.D.summary()

        return self.D

    def generator(self):
        # TODO : add generator topology
        if self.G:
            return self.G

        self.G = Sequential()
        dropout = 0.3
        filters = 128

        self.G.add(Dense(128*16*16, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(alpha=0.1))
        self.G.add(Reshape((int(self.img_rows/2), int(self.img_cols/2), filters)))
        
        # self.G.add(Dropout(dropout))

        self.G.add(Conv2D(filters, kernel_size = 5, padding='same', strides=1))
        self.G.add(LeakyReLU(alpha=0.1))
        self.G.add(BatchNormalization(momentum=0.9))

        # self.G.add(Dropout(dropout))

        #self.G.add(UpSampling2D(size=(2,2)))
        self.G.add(Conv2DTranspose(filters, 4, padding='same', strides=2))
        self.G.add(BatchNormalization(momentum=0.9))

        #self.G.add(Dropout(dropout))

        #self.G.add(UpSampling2D(size=(2,2)))
        self.G.add(Conv2D(filters, kernel_size = 5, padding='same', strides=1))
        self.G.add(LeakyReLU(alpha=0.1))
        self.G.add(BatchNormalization(momentum=0.9))

        self.G.add(Conv2D(filters, kernel_size = 5, padding='same', strides=1))
        self.G.add(LeakyReLU(alpha=0.1))
        self.G.add(BatchNormalization(momentum=0.9))

        self.G.add(Conv2D(int(filters/2), kernel_size = 3, padding='same', strides=1))
        self.G.add(LeakyReLU(alpha=0.1))
        self.G.add(BatchNormalization(momentum=0.9))

        self.G.add(Conv2D(int(filters/4), kernel_size = 3, padding='same', strides=1))
        self.G.add(LeakyReLU(alpha=0.1))
        self.G.add(BatchNormalization(momentum=0.9))

        # self.G.add(Dropout(dropout))

        self.G.add(Conv2D(self.channels, 5, padding='same', activation='tanh'))

        self.G.summary()

        return self.G

    def discriminator_model(self):
        # TODO : Compile the discriminator : choose Optimizer and loss function
        if self.DM:
            return self.DM

        optimizer = Adam(0.0002, 0.5)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return self.DM

    def adversarial_model(self):
        #TODO : Add generator + discriminator + choose optimizer and losse function
        if self.AM:
            return self.AM

        optimizer = Adam(0.0002, 0.5)
        disc = self.discriminator()
        disc.trainable = False
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(disc)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return self.AM
