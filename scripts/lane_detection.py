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
from os.path import expanduser

roi_x = 0
roi_y = 500
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

def perspective_warp(img, dst_size, src, dst):

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def sliding_window(img, nwindows=15, margin=40, minpix = 1, draw_windows=True):
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
    #prev_point = []

    for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = img.shape[0] - (window+1)*window_height
      win_y_high = img.shape[0] - window*window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin

      # Draw the windows on the visualization image
      if draw_windows == True:
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3)

      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
      (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
      (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)

      # If you found > minpix pixels, recenter next window on their mean position
      if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

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

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    #return out_img
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def inv_perspective_warp(img,
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[1]-1, img.shape[1])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    print points
    cv2.fillPoly(color_img, np.int_(points), (0,200,255))
    inv_perspective = inv_perspective_warp(color_img,dst_size=(img.shape[0],img.shape[1]))
    #inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[1]-1, img.shape[1])
    y_eval = np.max(ploty)
    ym_per_pix = 37.5/ img.shape[1] # meters per pixel in y dimension
    xm_per_pix = 3.5/ img.shape[1] # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)

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

    # Detect the object based on HSV Range Values
    mask = cv2.inRange(Roi, lower_t, upper_t)

    # Opening the image
    kernel = np.ones((7,7),np.uint8)
    eroded = cv2.erode(mask, kernel, iterations = 1) # eroding + dilating = opening
    wscale = cv2.dilate(eroded, kernel, iterations = 1)
    ret,thresh = cv2.threshold(wscale,128,255,cv2.THRESH_BINARY_INV) # thresholding the image //THRESH_BINARY_INV

    # Perspective warp
    rheight, rwidth = thresh.shape[:2]
    dst_size = (rheight,rwidth)
    dst = np.float32([(0,0), (1,0), (0,1), (1,1)])
    src = np.float32([(300,0), (1700,0), (300,rheight), (1700,rheight)]) # Find suitable Parameters

    warped_img = perspective_warp(thresh, dst_size, src, dst)

    # Sliding Window Search
    out_img, curves, lanes, ploty = sliding_window(warped_img)

    #plt.imshow(out_img)
    #plt.plot(curves[0], ploty, color='yellow', linewidth=1)
    #plt.plot(curves[1], ploty, color='yellow', linewidth=1)
    #plt.show()

    #curverad=get_curve(Roi, curves[0],curves[1])
    #print(curverad)

    #Roi_c = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    img_ = draw_lanes(thresh, curves[0], curves[1])

    #plt.imshow(img_, cmap='hsv')
    #plt.show()
    return img_

def lane_detector():
  rospy.init_node('lane_detector')

  home = expanduser("~/ICRA_2020/wheel_tracks.avi")
  cap = cv2.VideoCapture(home)

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
