import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle

# Read in car and non-car images
car_dir = './vehicles'
noncar_dir = './non-vehicles'
car_images = glob.glob(car_dir+'/**/*.png', recursive=True)
noncar_images = glob.glob(noncar_dir+'/**/*.png', recursive=True)
cars = []
notcars = []
for image in car_images+noncar_images:
    if image in noncar_images:
        notcars.append(image)
    else:
        cars.append(image)

from numpy import random
sample_size = 7000
car = cars[random.randint(len(cars))]
not_car = notcars[random.randint(len(notcars))]

i_car = mpimg.imread(car)
i_noncar = mpimg.imread(not_car)

fig, axes = plt.subplots(1,2, figsize=(4,2))
axes[0].set_title('car')
axes[0].imshow(i_car)
axes[1].set_title('non car')
axes[1].imshow(i_noncar)
plt.tight_layout()

plt.savefig('car_noncar.png')

