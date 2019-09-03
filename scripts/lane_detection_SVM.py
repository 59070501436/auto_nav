#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import math
import tf
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser
import matplotlib.pyplot as plt

class lane_finder_SVM():
    '''
    A class to find lane points given an image that has been inverse perspective mapped and scrubbed of most features
    other than the lanes.
    '''
    def __init__(self, image):
    #### Hyperparameters ####
        self.image = image

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs, color_space='RGB', hist_bins=32,
                         spatial_size=(32, 32),
                         orient=9, pix_per_cell=(8,8), cell_per_block=(2,2), hog_channel=0,
                         hist_feat=True, spatial_feat=True, hog_feat=True):
    '''
    imgs:  list of image filenames

    color_space: optional for color feature extraction,'RGB', 'HSV', 'LUV', 'HLS', 'YUV' or 'YCrCb'

    hist_bins: optional for color feature extraction, bins is an int, it defines the number of
               equal-width bins in the given range

    spatial_size: optional for spatial bining, 2-tuple, spatial binning output image size

    orient: optional for HOG feature extraction, integer, and represents the number of orientation bins
            that the gradient information will be split up into in the histogram. Typical values are
            between 6 and 12 bins

    pix_per_cell: optional 2-tuple, for HOG feature extraction, cell size over which each gradient histogram is computed.
                  This paramater is passed as a 2-tuple so you could have different cell sizes in x and y,
                  but cells are commonly chosen to be square

    cell_per_block: optional 2-tuple, for HOG feature extraction, specifies the local area over which the
                    histogram counts in a given cell will be normalized. Block normalization is not
                    necessarily required, but generally leads to a more robust feature set

    hog_channel: optional for HOG feature extraction, which channel t apply HOG: 0, 1, 2 or "ALL"

    hist_feat: flag to apply or not color histogram feature extraction

    spatial_feat: flag to apply or not spatial binning

    hog_feat: flag to apply or not HOG feature extraction
    '''
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # apply color conversion if other than 'RGB'
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)

            if spatial_feat == True:
                spatial_features = bin_spatial(feature_image, size=spatial_size)
                #print('len spatial_features in extract_features',len(spatial_features))
                file_features.append(spatial_features)
            if hist_feat == True:
                # Apply color_hist()
                hist_features = color_hist(feature_image, nbins=hist_bins)
                #print('len hist_features in extract_features',len(hist_features))
                file_features.append(hist_features)
            if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(get_hog_features(feature_image[:,:,channel],
                                            orient, pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                    #print('len hog_features in extract_features',len(hog_features))
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features

if __name__ == '__main__':

  try:
   rospy.init_node('horizon_detection', anonymous=True)

   # Load an color image in grayscale
   #home = expanduser("~/Third_Paper/SKP_post_harvest_dataset/Photos/_Color_603.png")
   #rgb_img = cv2.imread(home,0)
   crops = glob.glob('training_dataset/vehicles/**/*.png')
   lanes = glob.glob('training_dataset/non-vehicles/**/*.png')

   # Parameter tunning
   color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
   orient = 9  # HOG orientations
   pix_per_cell = (8,8) # HOG pixels per cell
   cell_per_block = (2,2) # HOG cells per block
   hog_channel = 0 # Can be 0, 1, 2, or "ALL"
   spatial_size = (32, 32) # Spatial binning dimensions
   hist_bins = 32    # Number of histogram bins
   spatial_feat = True # Spatial features on or off
   hist_feat = True # Histogram features on or off
   hog_feat = True # HOG features on or off
   y_start_stop = [None, None] # Min and max in y to search in slide_window()

   car_features = extract_features(cars, color_space=color_space,
                            hist_bins=hist_bins, spatial_size=spatial_size,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, hist_feat=hist_feat,
                            spatial_feat=spatial_feat, hog_feat=hog_feat)

   notcar_features = extract_features(notcars, color_space=color_space,
                            hist_bins=hist_bins, spatial_size=spatial_size,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, hist_feat=hist_feat,
                            spatial_feat=spatial_feat, hog_feat=hog_feat)

   while not rospy.is_shutdown():
     lf = lane_finder_SVM(rgb_img)

     # Closes all the frames
     #cv2.waitKey(0)
     #cv2.destroyAllWindows()

  except rospy.ROSInterruptException:
   pass
