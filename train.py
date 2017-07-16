import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from lesson_functions import *
#from hog_functions import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import sklearn
sklearn_version = float(sklearn.__version__)
if sklearn_version>=0.18:
    from sklearn.model_selection import train_test_split
else:
    from sklearn.cross_validation import train_test_split


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

# TODO play with these values to see how your classifier
# performs under different binning scenarios
spatial = 16
histbin = 16 # 32
colorspace = 'RGB2HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

print('n_cars:', len(cars))
print('n_noncars:', len(notcars))
import random
sample_size = 7000
cars = random.sample(cars, sample_size) #cars[0:sample_size]
notcars = random.sample(notcars, sample_size) #notcars[0:sample_size]

t=time.time()
car_features = extract_features(cars, cspace=colorspace, spatial_size=(spatial, spatial),
                        hist_bins=histbin, hist_range=(0, 256), orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, spatial_size=(spatial, spatial),
                        hist_bins=histbin, hist_range=(0, 256), orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
                        
#car_hog_features = extract_hog_features(cars, cspace=colorspace, orient=orient, 
#                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
#                        hog_channel=hog_channel)
#notcar_hog_features = extract_hog_features(notcars, cspace=colorspace, orient=orient, 
#                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
#                        hog_channel=hog_channel)
t2 = time.time()
print('Feature extraction:', round(t2-t, 2),  'seconds')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

'''
print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
'''

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

storage = { "svc": svc, 
            "scaler": X_scaler,
            "orient": orient,
            "pix_per_cell": pix_per_cell,
            "cell_per_block": cell_per_block,
            "spatial_size": (spatial, spatial),
            "hist_bins": histbin,
            "colorspace": colorspace,
            "hog_channel": hog_channel }
pickle.dump(storage, open( "storage.p", "wb" ) )
