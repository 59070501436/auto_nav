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
import matplotlib.pyplot as plt1

# for the lbp
from skimage import feature

numPoints = 24

class lane_finder_SVM():
    '''
    A class to find lane points given an image that has been inverse perspective mapped and scrubbed of most features
    other than the lanes.
    '''
    def __init__(self, image):
    #### Hyperparameters ####

        self.image = image

        # Gabor Filter Parameters
        # self.ksize = (24, 24)
        # self.sigma = 5
        self.theta = 0 #np.pi/4
        # self.lambda1 = 4
        self.gamma = 0.5
        #self.psi = 0
        self.ktype = cv2.CV_32F
        ksize = 24
        sigma = 2
        lambda1 = 4
        psi = 2

        # LBP Patterns
        # store the number of points and radius
        self.numPoints = 24 #default:24 #numPoints
        self.radius = 8 #default:8 #radius
        # initialize the local binary patterns descriptor along with
        # the data and label lists
        self.desc = LocalBinaryPatterns(24, 8)
        self.data = []
        self.labels = []

    def nothing_1(self, val):
        pass
    def nothing_2(self, val):
        pass
    def nothing_3(self, val):
        pass
    def nothing_4(self, val):
        pass
    def nothing_5(self, val):
        pass
    def nothing_6(self, val):
        pass

    # Our main driver function to return the segmentation of the input image.
    def TextureSegmentation(self, numPoints, radius):

        # # Gabor Filter
        # g_kernel = cv2.getGaborKernel((ksize,ksize), sigma, self.theta, lambda1, self.gamma, psi, self.ktype)
        # filtered_img = cv2.filter2D(self.image, cv2.CV_8UC3, g_kernel)

        # Linear Binary Patterns
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        features = feature.local_binary_pattern(gray, numPoints, radius, method="default") # method="uniform")

        return features 

if __name__ == '__main__':

  try:
   rospy.init_node('horizon_detection', anonymous=True)

   # Load an color image in grayscale
   home = expanduser("~/Third_Paper/Dataset_skp/overhead_img13.png") #SKP_post_harvest_dataset/Photos/SKP_6/left0000.jpg")
   # home = expanduser("~/Third_Paper/Frogn_Fields/frogn_fields_006.jpg") # Old Frogn Case
   # home = expanduser("~/Third_Paper/Frogn_Fields/Frogn_005/frogn_10000.png")

   rgb_img = cv2.imread(home)
   #rgb_img1 = cv2.imread(home1)

   # Getting ROI
   iheight, iwidth = rgb_img.shape[:2]
   Roi_s = rgb_img[0:int(iheight*0.4),0:iwidth]
   Roi_g = rgb_img[int(iheight*(1-0.4)):iheight,0:iwidth]
   Roi_m = rgb_img[int(iheight*0.4):int(iheight*(1-0.4)),0:iwidth]

   lf = lane_finder_SVM(Roi_m)
   #lf1 = lane_finder_SVM(rgb_img1)

   cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
   # cv2.namedWindow('preview1', cv2.WINDOW_NORMAL)

   # cv2.createTrackbar('ksize', 'preview' , 0, 50, lf.nothing_1)
   # cv2.createTrackbar('sigma', 'preview' , 0, 20, lf.nothing_2)
   # cv2.createTrackbar('theta', 'preview' , 0, 3, lf.nothing_3)
   # cv2.createTrackbar('lambda1', 'preview', 0, 20, lf.nothing_4)
   # cv2.createTrackbar('gamma', 'preview', 0, 1, lf.nothing_5)
   # cv2.createTrackbar('psi', 'preview', 0, 20, lf.nothing_6)

   cv2.createTrackbar('numPoints', 'preview', 0, 200, lf.nothing_5)
   cv2.createTrackbar('radius', 'preview', 0, 200, lf.nothing_6)
   # cv2.createTrackbar('numPoints', 'preview1', 0, 200, lf.nothing_5)
   # cv2.createTrackbar('radius', 'preview1', 0, 200, lf.nothing_6)

   while not rospy.is_shutdown():

      # # get current positions of four trackbars
      # ksize = cv2.getTrackbarPos('ksize','preview')
      # sigma = cv2.getTrackbarPos('sigma','preview')
      # theta = cv2.getTrackbarPos('theta','preview')
      # lambda1 = cv2.getTrackbarPos('lambda1','preview')
      # gamma = cv2.getTrackbarPos('gamma','preview')
      # psi = cv2.getTrackbarPos('psi','preview')
      # print ksize, sigma, lambda1, psi

      numPoints = cv2.getTrackbarPos('numPoints','preview')
      radius = cv2.getTrackbarPos('radius','preview')
      # numPoints1 = cv2.getTrackbarPos('numPoints','preview1')
      # radius1 = cv2.getTrackbarPos('radius','preview1')

      # ksize = 8
      # sigma = 4
      # theta = 0
      # lambda1 = 3
      # psi = 2
      # gamma = 0.5

      # numPoints = 6 #56
      # radius = 20 #172

      # img_texture = lf.TextureSegmentation(ksize, sigma, theta, lambda1, gamma, psi) # Function for texture filter
      img_texture = lf.TextureSegmentation(numPoints, radius) # Function for texture filter
      # img_texture1, hist1 = lf1.TextureSegmentation(numPoints1, radius1) # Function for texture filter
      # construct the figure
      # plt.style.use("ggplot")
      # (fig, ax) = plt.subplots()
      # fig.suptitle("Local Binary Patterns")
      # plt.ylabel("% of Pixels")
      # plt.xlabel("LBP pixel bucket")
      # ax.hist(img_texture.ravel(), normed=True, bins=64, range=(0, 64))
      # ax.set_xlim([0, 64])
      # ax.set_ylim([0, 0.035])
      # plt.show()
      # Closes all the frames
      # cv2.resizeWindow('preview', 800,800)
      cv2.imshow('preview', img_texture)
      k = cv2.waitKey(1) & 0xFF
      if k == 27:
        break
      # cv2.imshow('preview1', img_texture1)
      # k = cv2.waitKey(1) & 0xFF
      # if k == 27:
      #   break


   cv2.destroyAllWindows()
      #Filelocation = expanduser("~/filteredImages.png")
      #cv2.imwrite(Filelocation, filtered_img)

  except rospy.ROSInterruptException:
   pass
