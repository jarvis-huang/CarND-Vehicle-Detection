# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
### Writeup / README

[Link to the writeup document](https://github.com/jarvis-huang/CarND-Vehicle-Detection/blob/master/writeup.md) is the markdown writeup.
You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file `lession_functions.py` and `train.py`. Two functions `get_hog_features` and `extract_hog_features` implement the HOG extraction procedure.

I started by reading in all the `vehicle` and `non-vehicle` images. I selected 7000 images for training and remaining roughly 1800 images for validation. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I settled on `YCrCb` color space and the HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. This is because thye give the highest classification accuracy on the validation set. The validation accuracy was >0.99.

I tried various combinations of parameters and it turns out the combination of HOG, spatial and color histogram features gives the best results. I enabled all three feature types by seeting `hog_channel = "ALL"`. On Line 91 of `lesson_functions.py`, all three features are concatenanted to form a long feature vector.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using **LinearSVC** class from `sklearn.svm` module. Line 89~93 in `train.py` contains the training code.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is done inside the `find_car_bboxes` function of `main.py`. First I need to compute the number of blocks and steps to take in each direction. Then I do a double for loop to interate over all window locations to extract the features inside each window patch. Finally I run the SVM classifier over the feature vector to determine whether that patch contains a car. If a car is detected, the window location is recorded into a list.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

It is crucial to include multiple scales to increase detection rate. I used three scales (1.0, 1.5x, 2.0x) sliding windows and combined their outputs. However, this causes duplicate detection problem. I adopted the heatmap method, finding the pixels where most boxes intersect and threshold to get fewer boxes.

I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

To speed up computation, I compute the HOG features on the entire image all at once, and then slide a window across the feature map to extract smaller patches for classification. This is done is the `find_car_bboxes` function inside `main.py`.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

False positives is a serious issue in my detection result. Here's how I solve the problem. I used a combination of heatmap smoothing and temporal filtering to reduce the number of FPs. First, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from an image, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on that image:

### Here are the original image, heatmap, output of `scipy.ndimage.measurements.label()` and bounding boxes then overlaid on that image

![alt text][image5]

However, even after applying this technique the resultant bounding boxes are still very noisy from frame to frame and their shapes and locations change abruptly. This indicate that I need to apply some temporal filtering as well. First, I created a `Car` class to record various attributes of a detected car in the image, including centroid location, width and height, last detected frame and state. A car can have three different states: DETECTED, TRACKED and UNTRACKED. Initially a car is in UNTRACKED state and after a few consistent detections, it will transition to DETECTED state. Then we will keep a record of the number of consistent detections over N frames. If the number is higher than a threshold, we move to TRACKED state otherwise we downgrade to UNTRACKED. We only output bounding boxes when a car is in TRACKED state. For a detection to be associated with a car, its centroid and width and height must be close to previous values. In this way, we can make the tracking more robust and stable.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline will likely fail if there are many occlusions to cars. If a car is partly occluded, the detected features will be very different and the classifier will not be able to classify correctly. To mitigate this, we could enhance the tracker and use some other sensor data such as lidar or radar.

