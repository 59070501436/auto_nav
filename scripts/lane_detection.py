#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle
import math
from numpy import linalg as LA
from moviepy.editor import VideoFileClip

roi_x = 0
roi_y = 300
max_value_H = 360/2
max_value = 255
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = 145

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def perspective_warp(img,
                     dst_size=(1920,1080),
                     src=np.float32([(550.0,0.0),(1500,0),(1500,1023),(450,1034)]), # Choose the four vertices
                     dst=np.float32([(0,0), (1, 0), (1,1), (0,1)])):

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def sliding_window(img, nwindows=9, margin=75, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    # find peaks of left and right halves
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0) # Histrogram
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    left_lane_inds1= []
    right_lane_inds1 = []

    # empty lists to store the information of lane
    points = []
    confidenceValues = [1]

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        #if top < 0 or bottom > self.height - 1 or left < 0 or right > self.width - 1:
                #break

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        block_o= img[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
        whitePixels_o = np.argwhere(block_o == 255)

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        counter = 0
        step_size = 15
        leftx_current1 = leftx_current
        rightx_current1 = rightx_current
        whitePixelsvector_l = []
        whitePixelsvector_r = []

        # Each iteration is a fresh search attempt farther out from the last known lane point
        while (1):
            counter += 1
            leftx_current1 = leftx_current1+step_size
            rightx_current1 = rightx_current1+step_size
            # Break out if max iteartion exceeded
            if counter > 3:
             break
            win_y_low1 = img.shape[0] - (window+1)*window_height
            win_y_high1 = img.shape[0] - window*window_height
            win_xleft_low1 = leftx_current1 - margin
            win_xleft_high1 = leftx_current1 + margin
            win_xright_low1 = rightx_current1 - margin
            win_xright_high1 = rightx_current1 + margin

            # Identify the nonzero pixels in x and y within the window
            block_l = img[win_y_low1:win_y_high1, win_xleft_low1:win_xleft_high1]
            whitePixels_l = np.argwhere(block_l == 255)
            block_r = img[win_y_low1:win_y_high1, win_xright_low1:win_xright_high1]
            whitePixels_r = np.argwhere(block_r == 255)
            if whitePixels_l is not None:
              #print window, len(whitePixels_o), len(whitePixels) #avgPixel, newCenter, changeVector, rotatevvector
              whitePixelsvector_l.append(len(whitePixels_l))
            if whitePixels_r is not None:
              #print window, len(whitePixels_o), len(whitePixels_r) #avgPixel, newCenter, changeVector, rotatevvector
              whitePixelsvector_r.append(len(whitePixels_r))

        # Draw the windows on the visualization image
        #for whitePixels in whitePixelsvector:
        if len(whitePixelsvector_l)>0: #draw_windows == True
           #print np.max(whitePixelsvector)
           if np.max(whitePixelsvector_l) > len(whitePixels_o):
              cv2.rectangle(out_img,(win_xleft_low1,win_y_low1),(win_xleft_high1,win_y_high1),(0,0,255), 3)
              curr_win_xleft = [win_xleft_low1, win_xleft_high1]
           else:
              cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (100,255,255), 3)
              curr_win_xleft = [win_xleft_low1, win_xleft_high1]

        if len(whitePixelsvector_r)>0: #draw_windows == True
           if np.max(whitePixelsvector_r) > len(whitePixels_o):
              cv2.rectangle(out_img,(win_xright_low1,win_y_low1),(win_xright_high1,win_y_high1),(0,0,255), 3)
              curr_win_xright = [win_xright_low1, win_xright_high1]
           else:
              cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(100,255,255), 3)
              curr_win_xright = [win_xright_low1, win_xright_high1]

        # Identify the nonzero pixels in x and y within the window
        good_left_inds1 = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= curr_win_xleft[0]) &  (nonzerox < curr_win_xleft[1])).nonzero()[0]
        good_right_inds1 = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= curr_win_xright[0]) &  (nonzerox < curr_win_xright[1])).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds1.append(good_left_inds1)
        right_lane_inds1.append(good_right_inds1)

    # Concatenate the arrays of indices
    left_lane_inds1 = np.concatenate(left_lane_inds1)
    right_lane_inds1 = np.concatenate(right_lane_inds1)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds1]
    lefty = nonzeroy[left_lane_inds1]
    rightx = nonzerox[right_lane_inds1]
    righty = nonzeroy[right_lane_inds1]

    # Fit a second order polynomial to
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds1], nonzerox[left_lane_inds1]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds1], nonzerox[right_lane_inds1]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def vid_pipeline(cap):

   # Capture frame-by-frame
   ret, frame = cap.read()
   if ret == True:

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]

    # Getting ROI
    Roi = hsv[roi_y:height,roi_x:width] #-(roi_y+1),-(roi_x+1)

    # define range of blue color in HSV
    lower_t = np.array([low_H,low_S,low_V])
    upper_t = np.array([high_H,high_S,high_V])

    # Detect the object based on HSV Range Values v_min 71.65 v_max 179.0,242.25,200.60
    mask = cv2.inRange(Roi, lower_t, upper_t)

    # Opening the image
    kernel = np.ones((7,7),np.uint8)
    eroded = cv2.erode(mask, kernel, iterations = 1) # eroding + dilating = opening
    wscale = cv2.dilate(eroded, kernel, iterations = 1)
    ret,thresh = cv2.threshold(wscale,128,255,cv2.THRESH_BINARY_INV) # thresholding the image //THRESH_BINARY_INV

    # Perspective warp
    rheight, rwidth = thresh.shape[:2]
    warped_img = perspective_warp(thresh,dst_size=(rheight,rwidth),
                                 src=np.float32([(550.0,0.0),(1500,0),(1500,1023-(roi_y+1)),(450,1034-(roi_y+1))]))

    # Sliding Window Search
    out_img, curves, lanes, ploty = sliding_window(warped_img)

    return out_img

def lane_detector():
  rospy.init_node('lane_detector')

  #myclip = VideoFileClip('/home/saga/ICRA_2020/wheel_tracks.avi')#.subclip(40,43)
  cap = cv2.VideoCapture("/home/saga/ICRA_2020/wheel_tracks.avi")
  # Check if camera opened successfully
  if (cap.isOpened()== False):
   print("Error opening video stream or file")

  # Read until video is completed
  while(cap.isOpened()):

    while not rospy.is_shutdown():
      output = vid_pipeline(cap)
      # Display the resulting frame
      cv2.imshow('Frame',output)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
          break

      # Break the loop
      else:
          break

      # When everything done, release the video capture object
      cap.release()

      # Closes all the frames
      cv2.destroyAllWindows()
      #cv2.startWindowThread()
      #cv2.namedWindow("preview")
      #rospy.loginfo("aagadgv")

if __name__ == '__main__':
   try:
     lane_detector()
   except rospy.ROSInterruptException:
     pass
