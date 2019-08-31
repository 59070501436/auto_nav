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

    # def medianRGB(self, in_img):
    #    r_list = []
    #    g_list = []
    #    b_list = []
    #    PixelArray = np.asarray(in_img)
    #    for n, dim in enumerate(PixelArray):
    #         for num, row in enumerate(dim):
    #             r, g, b = row
    #             r_list.append(r)
    #             g_list.append(g)
    #             b_list.append(b)
    #
    #    r_median = np.median(r_list)
    #    g_median = np.median(g_list)
    #    b_median = np.median(b_list)
    #    return r_median, g_median, b_median

    def GenerateOverheadView(self, image):

        src=np.float32([(0,0), (1,0), (0,1), (1,1)])
        dst=np.float32([(0,0), (1,0), (0.2,1), (0.8,1)])

        img_size = np.float32([(image.shape[1],image.shape[0])])
        src = src* np.float32(img_size)
        dst_size =(image.shape[1],image.shape[1])
        dst = dst * np.float32(dst_size)

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped_img = cv2.warpPerspective(image, M, dst_size)
        return warped_img, M

    def EstimateRowDirection(self, src):

       # ## Iterate the image by varying angles
       #  # Skewing the image
       #  srcTri = np.array( [[0, 0], [src.shape[1] - 1, src.shape[0] - 1], [src.shape[1] - 1, 0] ] ).astype(np.float32)
       #  dstTri = np.array( [[0, 0], [(src.shape[1] - 1)*0.95, (src.shape[0] - 1)], [(src.shape[1] - 1)*0.95, 0] ] ).astype(np.float32)
       #
       #  warp_mat = cv2.getAffineTransform(srcTri, dstTri)
       #  print warp_mat
       #  warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

        # # Rotating the image after Warp
        # center = (src.shape[1]//2, src.shape[0]//2)
        # angle = 50
        # scale = 0
        # rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        # warp_rotate_dst = cv2.warpAffine(src, rot_mat, (src.shape[1], src.shape[0]))

        rows,cols = src.shape

        M = cv2.getRotationMatrix2D((cols/2,rows/2),5,1)
        dst = cv2.warpAffine(src,M,(cols,rows))

        # Sum up the columns
        img_col_sum = np.sum(dst, axis=0)
        x_coordinates = np.arange(cols)
        #var_arr = np.var(img_col_sum)
        #print var_arr

        # Calculate the variance
        #fig, ax = plt.subplots()
        plt.imshow(dst)
        #ax.plot(x_coordinates, img_col_sum)  #[img_col_sum])
        #ax.plot(x_coordinates, img_col_sum, '--', linewidth=5, color='firebrick')

        plt.plot(x_coordinates, img_col_sum, '--', linewidth=5, color='firebrick')
        plt.xlabel('x')
        plt.ylabel('s(x)')
        plt.show()

        # Find the angle with max variance
        heading_angle = 0

        return dst, heading_angle
if __name__ == '__main__':

  try:
   rospy.init_node('horizon_detection', anonymous=True)

   # Load an color image in grayscale
   home = expanduser("~/Third_Paper/SKP_post_harvest_dataset/Photos/_Color_603.png")
   rgb_img = cv2.imread(home,0)

   # Getting ROI
   iheight, iwidth = rgb_img.shape[:2]
   Roi_s = rgb_img[0:int(iheight*0.3),0:iwidth]
   Roi_g = rgb_img[int(iheight*(1-0.3)):iheight,0:iwidth]
   Roi_m = rgb_img[int(iheight*0.3):int(iheight*(1-0.3)),0:iwidth]

   while not rospy.is_shutdown():
     lf = lane_finder_SVM(rgb_img)

     overhead_img, M = lf.GenerateOverheadView(Roi_m)

     skewed_img, skew_angle = lf.EstimateRowDirection(overhead_img)

     # cv2.startWindowThread()
     # cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
     # cv2.resizeWindow('preview', 800,800)
     # cv2.imshow('preview', skewed_img)
     #cv2.imwrite("/home/saga/skewed_img.png", skewed_img)

     # r_ms, g_ms, b_ms = lf.medianRGB(Roi_sky)
     # r_mg, g_mg, b_mg = lf.medianRGB(Roi_ground)
     #
     # rheight, rwidth = img.shape[:2]
     # b, g, r = cv2.split(Roi_m)
     # dist = np.linalg.norm((b,g,r)-(b_mg[0],g_mg[0],r_mg[0]))
     # print dist

     # Closes all the frames
     cv2.waitKey(0)
     cv2.destroyAllWindows()

  except rospy.ROSInterruptException:
   pass
