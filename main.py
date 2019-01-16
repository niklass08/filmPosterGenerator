import numpy as np
from random import randint
import matplotlib.pyplot as plt

from filmPosterGan import filmPosterGan
from gan import Gan

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

g = Gan(64,64,3)
g.discriminator()
g.generator()
