import numpy as np
from random import randint
import matplotlib.pyplot as plt

from filmPosterGan import filmPosterGan
from gan import Gan

grayscale = False
channels = 1 if grayscale else 3
fpg = filmPosterGan(rows = 64, cols = 64, channels = channels, dataFolder = "./data")
data_gen = fpg.load_data(grayscale=grayscale)
print(data_gen)
(x, y) = data_gen.next()
print(x.shape)

i = randint(1,3)
print(x[i].shape)
image = x[i]
if grayscale: image = np.reshape(image, (64, 64))
print("Dimension of input image=",np.shape(image))

#plt.imshow(image.transpose(2,1,0))
if i>0: plt.figure()
if grayscale:
    plt.imshow(image, cmap='gray')
else:
    plt.imshow(image.astype('uint8'))
plt.show()


fpg.train(train_steps=2000, batch_size=16, save_interval=100)
