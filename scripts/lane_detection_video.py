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
import tf
from numpy import linalg as LA
from moviepy.editor import VideoFileClip
from os.path import expanduser
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import CameraInfo

K = CameraInfo()
cam_param_receive = False
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

def imagecaminfoCallback(data):
    global cam_param_receive, K
    K = data.K
    cam_param_receive = True

def perspective_warp(img, dst_size, src, dst): # Choose the four vertices

    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    #dst_size1 = (1920,580)
    #unwarped = cv2.warpPerspective(warped, Minv, dst_size1)

    return warped, M #, Minv, unwarped

def inv_perspective_warp(img, dst_size, src, dst):
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
    return warped, M

def sliding_window(img, nwindows=15, margin=50, minpix=1, draw_windows=True):
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

    for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = img.shape[0] - (window+1)*window_height
      win_y_high = img.shape[0] - window*window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin

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

      # Draw the windows on the visualization image
      if draw_windows == True:
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (255,0,0), 3)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (255,0,0), 3)

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
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 100, 255] #[255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def vid_pipeline(img_frame):

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
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
    ret,thresh = cv2.threshold(wscale, 128, 255, cv2.THRESH_BINARY_INV) # thresholding the image //THRESH_BINARY_INV

    # Perspective warp
    rheight, rwidth = thresh.shape[:2]
    dst_size =(rheight,rwidth)
    src=np.float32([(0.05,0), (1,0), (0.05,1), (1,1)])
    dst=np.float32([(0,0), (1,0), (0,1), (1,1)])
    warped_img, M  = perspective_warp(thresh, dst_size, src, dst)

    # Sliding Window Search
    out_img, curves, lanes, ploty = sliding_window(warped_img)

    # Fitted curves as points
    leftLane = np.array([np.transpose(np.vstack([curves[0], ploty]))])
    rightLane = np.array([np.flipud(np.transpose(np.vstack([curves[1], ploty])))])
    points = np.hstack((leftLane, rightLane))
    curves_m = (curves[0]+curves[1])/2
    midLane = np.array([np.transpose(np.vstack([curves_m, ploty]))])

    leftLane1 = leftLane[0].astype(int)
    rightLane1 = rightLane[0].astype(int)
    midLane1 = midLane[0].astype(int)

    cv2.polylines(out_img, [leftLane1], 0, (0,255,255), thickness=5, lineType=8, shift=0)
    cv2.polylines(out_img, [rightLane1], 0, (0,255,255), thickness=5, lineType=8, shift=0)
    cv2.polylines(out_img, [midLane1], 0, (255,0,255), thickness=5, lineType=8, shift=0)

    dst_size =(rwidth, rheight)
    invwarp, Minv = inv_perspective_warp(out_img, dst_size, dst, src)

    # Combine the result with the original image
    img_frame[roi_y:height,roi_x:width] = cv2.addWeighted(img_frame[roi_y:height,roi_x:width],
                                                                       1, invwarp, 0.9, 0)
    result = img_frame

    return midLane1, out_img, result #invwarp

def camera2world(x_c, t_c, R_c):
 # ray in world coordinates
 #x_c_q = np.array([0,x_c[0],x_c[1],x_c[2]])
 #x_wq_o = np.matmul(R_c, x_c_q)

 # R_c1 = np.array([[R_c[0]], [R_c[1]], [R_c[2]], [R_c[3]]])
 x_c_q = np.array([[0.0], [x_c[0]], [x_c[1]], [x_c[2]]])

 x_wq_o = np.multiply(R_c,x_c_q)
 x_wq = np.multiply(x_wq_o,np.conj(R_c))
 x_w = np.array([x_wq[1],x_wq[2],x_wq[3]])

 # distance to the plane
 ## d = dot((t_p - t_c),n_p)/dot(x_w,n_p)
 ## simplified expression assuming plane t_p = [0 0 0]; n_p = [0 0 1];
 d = -t_c[2]/x_w[2]

 # intersection point
 x_p = np.add(x_w,t_c)

 #print x_p

 return x_p

def lane_detector():
  topic = 'test_poses'
  publisher = rospy.Publisher(topic, PoseArray)
  rospy.init_node('lane_detector_video', anonymous=True)

  rospy.Subscriber("/kinect2_camera/rgb/camera_info", CameraInfo, imagecaminfoCallback)

  #rospy.spin()
  listener = tf.TransformListener()

  home = expanduser("~/ICRA_2020/wheel_tracks_ct.avi")
  cap = cv2.VideoCapture(home)

  # Check if camera opened successfully
  if (cap.isOpened()== False):
   print("Error opening video stream or file")

  # Read until video is completed
  while(cap.isOpened()):

   # Capture frame-by-frame
   ret, frame = cap.read()
   if ret == True:

    while not rospy.is_shutdown():

      #centerLine, warp_img, output = vid_pipeline(frame)

      # Camera Parmeters
      # Calcuate 3D World Point from 2D Image Point
      #p_c = np.array([centerLine[0][0], centerLine[0][1], 1])
      p_c = np.array([0, 0, 1])
      global cam_param_receive, K

      if cam_param_receive==True:
        K_arr = list(K) # Convert tuple into array
        K_f = np.reshape(K_arr, (3, 3)) # Resize array as 3*3 matrix
        K_inv = np.linalg.inv(K_f)
        x_c = np.matmul(K_inv, p_c)

        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", K[0])
        #cam_param_receive = False

          #try:
             #(trans,rot) = listener.lookupTransform('map', 'kinect2_rgb_optical_frame', rospy.Time(0))
          #except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
             #continue

        trans = np.array([0.800, -0.010, 1.750])
        rot = np.array([0.683, -0.683, 0.183, -0.184])

        t_c = np.array([[trans[0]], [trans[1]], [trans[2]]])
        R_c = np.array([[rot[0]], [rot[1]], [rot[2]], [rot[3]]])
        x_p = camera2world(x_c, t_c, R_c)
        print x_p


        # # Used to publish waypoints as pose array so that you can see them in rviz, etc.
        poses = PoseArray()
        poses.header.frame_id = "map"
        poses.header.stamp = rospy.Time.now()

        pose = Pose()
        pose.position.x = x_p[1]
        pose.position.y = x_p[0]
        pose.position.z = 0 #x_p[2]
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1
        poses.poses.append(pose)

        publisher.publish(poses)

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

      # Display the resulting frame
      #cv2.startWindowThread()
      #cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
      #cv2.resizeWindow('preview', 800,800)
      #cv2.imshow('preview', warp_img)

      #fheight, fwidth = output.shape[:2]
      #print warp_img.shape, output.shape
      #warp_img = cv2.resize(warp_img,(int(fwidth),int(fheight)))
      #numpy_horizontal = np.hstack((warp_img, output))

      #cv2.namedWindow('preview1', cv2.WINDOW_NORMAL)
      #cv2.resizeWindow('preview1', 800,800)
      #cv2.imshow('preview1', output)

      # Plotting the data
      # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
      # ax1.set_title('Original', fontsize=10)
      # ax1.xaxis.set_visible(False)
      # ax1.yaxis.set_visible(False)
      # ax1.imshow(frame, aspect="auto")
      # camera = Camera(f)
      # ax2.set_title('Filter+Perspective Tform', fontsize=10)
      # ax2.xaxis.set_visible(False)
      # ax2.yaxis.set_visible(False)
      # ax2.imshow(warped_img, aspect="auto")
      #
      # ax3.plot(curves[0], ploty, color='yellow', linewidth=5)
      # ax3.plot(curves[1], ploty, color='yellow', linewidth=5)
      # ax3.xaxis.set_visible(False)
      # ax3.yaxis.set_visible(False)
      # ax3.set_title('Sliding window+Curve Fit', fontsize=10)
      # ax3.imshow(out_img, aspect="auto")
      #
      # ax4.set_title('Overlay Lanes', fontsize=10)
      # ax4.xaxis.set_visible(False)
      # ax4.yaxis.set_visible(False)
      # ax4.imshow(result, aspect="auto")
      #
      # # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
      # animation.FuncAnimation(f, ax1, interval=2, blit=True)
      # animation = camera.animate()
      # plt.show()
      rospy.sleep(1)  # sleep for one second

if __name__ == '__main__':
   try:
     lane_detector()
   except rospy.ROSInterruptException:
     pass
