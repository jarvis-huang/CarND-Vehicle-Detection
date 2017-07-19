import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import pickle
import cv2

ig, axes = plt.subplots(2,2, figsize=(9,6))
axes[0,0].imshow(mpimg.imread('output_images/test1_out.png'))
axes[0,1].imshow(mpimg.imread('output_images/test3_out.png'))
axes[1,0].imshow(mpimg.imread('output_images/test4_out.png'))
axes[1,1].imshow(mpimg.imread('output_images/test5_out.png'))
plt.tight_layout()
plt.savefig('output_images/sample_output.png')
