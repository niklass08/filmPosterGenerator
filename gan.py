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
        self.channels = channels
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        # TODO : add discriminator topology
        if self.D:
            return self.D

        self.D = Sequential()
        dropout = 0.5
        output_size = 32
        input_shape = (self.img_rows, self.img_cols, self.channels)
        self.D.add(Conv2D(output_size, 5, padding='same', data_format='channels_last', input_shape=input_shape, strides=2, activation='relu'))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(output_size*2, 5, padding='same', strides=2, activation='relu'))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(output_size*4, 5, padding='same', strides=2, activation='relu'))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(output_size*8, 5, padding='same', strides=1, activation='relu'))
        self.D.add(Dropout(dropout))

        self.D.add(Flatten())
        self.D.add(Dense(1, activation='sigmoid'))

        self.D.summary()

        return self.D

    def generator(self):
        # TODO : add generator topology
        return self.G

    def discriminator_model(self):
        # TODO : Compile the discriminator : choose Optimizer and loss function
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        #TODO : Add generator + discriminator + choose optimizer and losse function
        return self.AM
