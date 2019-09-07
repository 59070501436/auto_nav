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

        # Gabor Filter Parameters
        # self.ksize = (24, 24)
        # self.sigma = 5
        self.theta = np.pi/4 #np.pi/4
        # self.lambda1 = 4
        self.gamma = 0.5
        #self.psi = 0
        self.ktype = cv2.CV_32F

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
    def runGabor(self, ksize, sigma, theta, lambda1, gamma, psi):

        g_kernel = cv2.getGaborKernel((ksize,ksize), sigma, self.theta, lambda1, self.gamma, psi, self.ktype)
        filtered_img = cv2.filter2D(self.image, cv2.CV_8UC3, g_kernel)
        return filtered_img

if __name__ == '__main__':

  try:
   rospy.init_node('horizon_detection', anonymous=True)

   # Load an color image in grayscale
   #home = expanduser("~/Third_Paper/SKP/overhead_img.png") #SKP_post_harvest_dataset/Photos/SKP_6/left0000.jpg")
   home = expanduser("~/Third_Paper/Datasets/Frogn/frogn_001.jpg") #SKP_post_harvest_dataset/Photos/SKP_6/left0000.jpg")
   rgb_img = cv2.imread(home)

   lf = lane_finder_SVM(rgb_img)
   cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
   cv2.createTrackbar('ksize', 'preview' , 0, 50, lf.nothing_1)
   cv2.createTrackbar('sigma', 'preview' , 0, 20, lf.nothing_2)
   cv2.createTrackbar('theta', 'preview' , 0, 3, lf.nothing_3)
   cv2.createTrackbar('lambda1', 'preview', 0, 20, lf.nothing_4)
   cv2.createTrackbar('gamma', 'preview', 0, 1, lf.nothing_5)
   cv2.createTrackbar('psi', 'preview', 0, 20, lf.nothing_6)

   while not rospy.is_shutdown():

      # get current positions of four trackbars
      ksize = cv2.getTrackbarPos('ksize','preview')
      sigma = cv2.getTrackbarPos('sigma','preview')
      theta = cv2.getTrackbarPos('theta','preview')
      lambda1 = cv2.getTrackbarPos('lambda1','preview')
      gamma = cv2.getTrackbarPos('gamma','preview')
      psi = cv2.getTrackbarPos('psi','preview')
      print ksize, sigma, lambda1, psi

      img_texture = lf.runGabor(ksize,sigma,theta,lambda1,gamma,psi) # Function for texture filter

      # Closes all the frames
      # cv2.resizeWindow('preview', 800,800)
      cv2.imshow('preview', img_texture)
      k = cv2.waitKey(1) & 0xFF
      if k == 27:
        break

   cv2.destroyAllWindows()
      #Filelocation = expanduser("~/filteredImages.png")
      #cv2.imwrite(Filelocation, filtered_img)

  except rospy.ROSInterruptException:
   pass
