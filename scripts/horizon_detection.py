#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import math
import tf
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser

def medianRGB(in_img):
   r_list = []
   g_list = []
   b_list = []
   PixelArray = np.asarray(in_img)
   for n, dim in enumerate(PixelArray):
        for num, row in enumerate(dim):
            r, g, b = row
            r_list.append(r)
            g_list.append(g)
            b_list.append(b)

   r_median = np.median(r_list)
   g_median = np.median(g_list)
   b_median = np.median(b_list)
   return r_median, g_median, b_median

if __name__ == '__main__':

  try:
   rospy.init_node('horizon_detection', anonymous=True)

   # Load an color image in grayscale
   home = expanduser("~/Third_Paper/Frogn_Fields/frogn_fields_001.jpg")
   img = cv2.imread(home)

   # Getting ROI
   iheight, iwidth = img.shape[:2]
   Roi_sky = img[0:int(iheight*0.3),0:iwidth]
   r_ms, g_ms, b_ms = medianRGB(Roi_sky)
   Roi_ground = img[int(iheight*(1-0.3)):iheight,0:iwidth]
   r_mg, g_mg, b_mg = medianRGB(Roi_ground)

   Roi_m = img[int(iheight*0.3):int(iheight*(1-0.3)),0:iwidth]
   rheight, rwidth = img.shape[:2]
   b, g, r = cv2.split(Roi_m)
   dist = np.linalg.norm((b,g,r)-(b_mg[0],g_mg[0],r_mg[0]))
   print dist

   #print  r-r_mg, g-g_mg, b-b_mg #r-r_ms, g-g_ms, b-b_ms,
   #math.sqrt(pow(r-r_mg,2) + pow(g-g_mg,2) + pow(b-b_mg,2)) #r-r_ms, g-g_ms, b-b_ms,
   # for i in range(rheight):
   #   for j in range(rwidth):
   #       Roi_m[i,j]

  except rospy.ROSInterruptException:
   pass
