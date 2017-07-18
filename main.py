import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import pickle
import cv2
from lesson_functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip # needed to edit/save/watch video clips
from Car import *

dist_pickle = pickle.load( open("storage.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
colorspace = dist_pickle["colorspace"]
hog_channel = dist_pickle["hog_channel"]

#img = mpimg.imread('test_images/test1.jpg')
#img = mpimg.imread('test_images/test3.jpg')
#img = mpimg.imread('test_images/test4.jpg')
#img = mpimg.imread('test_images/test5.jpg')
#img = mpimg.imread('test_images/test6.jpg')

#img = mpimg.imread('web/1.jpg')
#img = mpimg.imread('web/2.jpg')
#img = mpimg.imread('web/3.jpg')
#img = cv2.resize(img, (1280,720))

def get_heat_map(img, bboxes):
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    for bb in bboxes:
        top_left, bottom_right = bb[0], bb[1]
        xmin, ymin, xmax, ymax = top_left[0], top_left[1], bottom_right[0], bottom_right[1]
        heatmap[ymin:ymax,xmin:xmax] += 1
    return heatmap

def find_bboxes_from_heatmap(heatmap, thresh=2):
    heatmap[heatmap<thresh] = 0
    labels = label(heatmap)
    bboxes = []
    for car_num in range(1, labels[1]+1):
        nonzero = (labels[0]==car_num).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bboxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
    return bboxes

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_car_bboxes(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    bboxes = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv=colorspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel=="ALL":
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_feats = [hog_feat1, hog_feat2, hog_feat3]
                hog_features = np.hstack((hog_feats[hog_channel]))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_feat)).reshape(1, -1))    
            #test_features = X_scaler.transform(hog_features.reshape(1, -1))  
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return bboxes
  
def draw_bboxes(img, bboxes):
    draw_img = np.copy(img)
    for bb in bboxes:
        cv2.rectangle(draw_img,bb[0],bb[1],(0,0,255),3)
    return draw_img
    
def is_bb_overlap(bb1, bb2):
    xmin1, xmax1, ymin1, ymax1 = bb1[0][0], bb1[1][0], bb1[0][1], bb1[1][1]
    xmin2, xmax2, ymin2, ymax2 = bb2[0][0], bb2[1][0], bb2[0][1], bb2[1][1]
    
    if xmin1>=xmax2 or xmin2>=xmax1 or ymin1>=ymax2 or ymin2>=ymax1:
        return False
    else:
        return True

def get_overlapping_centroids(img, bboxes):
    # Group into overlapping groups
    bb_groups = []
    for bb in bboxes:
        assigned = False
        if not bb_groups:
            bb_groups.append([bb])
            continue
        else:
            for bb_list in bb_groups:
                belong = False
                for b in bb_list:
                    if is_bb_overlap(bb, b):
                        belong = True
                        break
                if belong:
                    bb_list.append(bb)
                    assigned = True
                    break
        if not assigned:
            bb_groups.append([bb])
        
    centroids = []
    for bb_list in bb_groups:
        c_x, c_y = 0, 0
        for bb in bb_list:
            c_x += (bb[0][0]+bb[1][0])/2.0
            c_y += (bb[0][1]+bb[1][1])/2.0
        c_x /= len(bb_list)
        c_y /= len(bb_list)
        centroids.append((int(c_x),int(c_y)))
        
    return centroids
   
def assign_new_centroids(cars, centroids, thresh=10):
    assigned = [False] * len(centroids)
    for car in cars:
        min_dist = -1
        nearest_c_idx = -1
        for i, cent in enumerate(centroids):
            if assigned[i]: continue
            dist = numpy.linalg.norm(np.array(car.centroid)-np.array(cent))
            if (min_dist<0 or dist<min_dist):
                min_dist = dist
                nearest_c_idx = i
        if min_dist>=0 and min_dist<=thresh: # assign nearest_c to car
            car.centroid = centroids[nearest_c_idx]
            car.age = 0
            car.M += 1
            #print("Assigned: ({},{})".format(car.centroid[0],car.centroid[1]))
            assigned[nearest_c_idx] = True
            
    # Create cars from non-assigned centroids
    #for c in [centroids[i] for i in range(len(centroids)) if not assigned[i]]:
    for c in [ci for ci,ai in zip(centroids, assigned) if not ai]:
        car = Car()
        car.centroid = c
        car.id = len(cars)
        #print("Created: ({},{})".format(c[0],c[1]))
        cars.append(car)
    
    # Increment age for all cars
    for car in cars:
        car.transition()
    
def assign_new_bbs(cars, bbs, dist_thresh=20, wl_thresh=10):
    assigned = [False] * len(bbs)
    for car in cars:
        min_dist = -1
        nearest_idx = -1
        nearest_centroid = ()
        nearest_bb = ()
        for i, bb in enumerate(bbs):
            if assigned[i]: continue
            cent = ((bb[0][0]+bb[1][0])//2, (bb[0][1]+bb[1][1])//2)
            dist = numpy.linalg.norm(np.array(car.centroid)-np.array(cent))
            if (min_dist<0 or dist<min_dist):
                min_dist = dist
                nearest_idx = i
                nearest_centroid = cent
                nearest_bb = bb
        if min_dist>=0 and min_dist<=dist_thresh and \
            np.fabs(nearest_bb[1][0]-nearest_bb[0][0]-car.w)<wl_thresh and \
            np.fabs(nearest_bb[1][1]-nearest_bb[0][1]-car.h)<wl_thresh: # assign nearest_c to car
            car.centroid = nearest_centroid
            car.w = nearest_bb[1][0]-nearest_bb[0][0]
            car.h = nearest_bb[1][1]-nearest_bb[0][1]
            car.age = 0
            car.M += 1
            #print("Assigned: ({},{})".format(car.centroid[0],car.centroid[1]))
            assigned[nearest_idx] = True
            
    # Create cars from non-assigned centroids
    for bb in [bb for bb,ass in zip(bbs, assigned) if not ass]:
        car = Car()
        car.centroid = ((bb[0][0]+bb[1][0])//2, (bb[0][1]+bb[1][1])//2)
        car.w = bb[1][0] - bb[0][0]
        car.h = bb[1][1] - bb[0][1]
        car.id = len(cars)
        #print("Created: ({},{})".format(c[0],c[1]))
        cars.append(car)
    
    # Increment age for all cars
    for car in cars:
        car.transition()
    
# Draw detected car centroids
def draw_cars(img, cars):
    out_img = np.copy(img)
    for car in [x for x in cars if x.state==CarState.TRACKED]:
        cv2.circle(out_img, car.centroid, 5, (255,0,0), thickness=-1)
        if car.w>0 and car.h>0:
            topleft = (car.centroid[0]-car.w//2, car.centroid[1]-car.h//2)
            bottomright = (car.centroid[0]+car.w//2, car.centroid[1]+car.h//2)
            cv2.rectangle(out_img, topleft, bottomright, (255,0,0), 3)
    return out_img


ystart = 400
ystop = 656
scale = 1.0
cars = []

# Order: RGB
def process_image(img):
    global ystart, ystop, scale, cars
    print("---")
    bboxes1 = find_car_bboxes(img, ystart, ystop, 1.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    bboxes2 = find_car_bboxes(img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    bboxes3 = find_car_bboxes(img, ystart, ystop, 2.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)    
    bboxes = bboxes1 + bboxes2 + bboxes3
    
    '''
    Method 1: Heatmap
    heatmap = get_heat_map(img, bboxes)
    bboxes_nms = find_bboxes_from_heatmap(heatmap, thresh=1) # non-max suppression
    out_img = draw_bboxes(img, bboxes_nms)
    heatmap = np.clip(heatmap, 0, 255)
    #mpimg.imsave("heatmap.png", heatmap, cmap='hot')
    #mpimg.imsave("detection.png", out_img)
    #plt.imshow(out_img)
    '''
    
    '''
    ## Method 2: centroid of overlapping bboxes + tracking
    centroids = get_overlapping_centroids(img, bboxes)
    assign_new_centroids(cars, centroids, thresh=20)
    out_img = draw_cars(img, cars)
    #for car in cars:
    #    print("car:", car.id, car.centroid, car.state)
    '''
    
    ## Method 3: Heatmap + centroid + tracking
    heatmap = get_heat_map(img, bboxes)
    bboxes_nms = find_bboxes_from_heatmap(heatmap, thresh=3) # non-max suppression
    centroids = []
    assign_new_bbs(cars, bboxes_nms, dist_thresh=30, wl_thresh=20)
    out_img = draw_cars(img, cars)
    
    
    return out_img
    
    
video_in = 'project_video.mp4'
clip = VideoFileClip(video_in)
#clip = VideoFileClip(video_in).subclip(10,20)
#clip = VideoFileClip(video_in).subclip(30,40)

#video_in = 'test_video.mp4'
#clip = VideoFileClip(video_in)

video_out = video_in.replace('.mp4', '_out.mp4')


new_clip = clip.fl_image(process_image)
new_clip.write_videofile(video_out, audio=False)
