import numpy as np
from random import randint
import matplotlib.pyplot as plt

from filmPosterGan import filmPosterGan
from gan import Gan

grayscale = False
fpg = filmPosterGan(rows = 28, cols = 28, channels = 3, dataFolder = "./data")
data_gen = fpg.load_data(grayscale=False)
print(data_gen)
(x, y) = data_gen.next()
print(x.shape)

i = randint(1,7)
print(x[i].shape)
image = x[i]
if grayscale: image = np.reshape(image, (28, 28))
print("Dimension of input image=",np.shape(image))

#plt.imshow(image.transpose(2,1,0))
if i>0: plt.figure()
if grayscale:
    plt.imshow(image, cmap='gray')
else:
    plt.imshow(image.astype('uint8'))
plt.show()


fpg.train(train_steps=2000, batch_size=32)
