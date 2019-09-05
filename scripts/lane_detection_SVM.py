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
        self.ksize = (24, 24)
        self.sigma = 5
        self.theta = np.pi/4 #np.pi/4
        self.lambda1 = 4
        self.gamma = 0.25
        self.psi = 0
        self.ktype = cv2.CV_32F
        self.g_kernel = cv2.getGaborKernel(self.ksize, self.sigma, self.theta, self.lambda1, self.gamma, self.psi, self.ktype)

    # Our main driver function to return the segmentation of the input image.
    def runGabor(self):
        filtered_img = cv2.filter2D(self.image, cv2.CV_8UC3, self.g_kernel)
        Filelocation = expanduser("~/filteredImages1.png")
        cv2.imwrite(Filelocation, filtered_img)

if __name__ == '__main__':

  try:
   rospy.init_node('horizon_detection', anonymous=True)

   # Load an color image in grayscale
   home = expanduser("~/Third_Paper/Dataset_skp/overhead_img.png") #SKP_post_harvest_dataset/Photos/SKP_6/left0000.jpg")
   rgb_img = cv2.imread(home)


   while not rospy.is_shutdown():
      lf = lane_finder_SVM(rgb_img)
      lf.runGabor()
     # Closes all the frames
     #cv2.waitKey(0)
     #cv2.destroyAllWindows()

  except rospy.ROSInterruptException:
   pass
