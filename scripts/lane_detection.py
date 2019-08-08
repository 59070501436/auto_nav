#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import numpy as np
import pandas as pd
import cv2
import os
import sys
import roslib
import matplotlib.pyplot as plt
import pickle
import math
import tf
from numpy import linalg as LA
from os.path import expanduser
import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseArray,Point
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import tf2_ros
import quaternion
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import KMeans
from itertools import imap
#from std_srvs.msg import Empty

K = CameraInfo()
rgb_img = Image()
#emptymsg = Empty()
cam_param_receive = False
img_receive = False
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

left_a, left_b, left_c, left_d = [],[],[],[]
right_a, right_b, right_c = [],[],[]

def imagecaminfoCallback(data):
    global cam_param_receive, K
    K = data.K
    cam_param_receive = True

def imageCallback(ros_data):
    global rgb_img, img_receive, bridge
    bridge = CvBridge()

    try:
      rgb_img = bridge.imgmsg_to_cv2(ros_data, "bgr8")
    except CvBridgeError as e:
      print(e)

    img_receive = True

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
    #print M
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped, M

def camera2world(x_c, t_c, R_c):

 # ray in world coordinates
 x_c_q = np.quaternion(0, x_c[0], x_c[1], x_c[2])
 x_wq = R_c*x_c_q*R_c.conjugate()
 x_w = np.array([x_wq.x,x_wq.y,x_wq.z])

 # distance to the plane
 ## d = dot((t_p - t_c),n_p)/dot(x_w,n_p)
 ## simplified expression assuming plane t_p = [0 0 0]; n_p = [0 0 1];
 d = -t_c[2]/x_w[2]

 # intersection point
 x_wd = np.array([(x_w[0]*d),(x_w[1]*d),(x_w[2]*d)])
 x_p = np.add(x_wd, t_c)

 return x_p

def normalizeangle(bearing): # Normalize the bearing

   if (bearing < -math.pi):
          bearing += 2 * math.pi
   elif (bearing > math.pi):
          bearing -= 2 * math.pi
   return bearing

def sliding_window(img, nwindows=15, margin=50, minpix=1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c, left_d
    left_fit_= np.empty(4) #3
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    #Finds the expected starting points  using K-Means
    clusters = 2

    # Crop the search space
    base_size = .1 # random number
    bottom = (img.shape[0] - int(base_size * img.shape[0]))
    base = img[bottom:img.shape[0], 0:img.shape[1]]

    # Find white pixels
    whitePixels = np.argwhere(base == 255)

    # Attempt to run kmeans (the kmeans parameters were not chosen with any sort of hard/soft optimization)
    try:
        kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=3, max_iter=150).fit(whitePixels)
    except:
    ## If kmeans fails increase the search space unless it is the whole image, then it fails
         if base_size > 1:
             return None
         else:
             base_size = base_size * 1.5
    #         initialPoints(clusters)
    # conver centers to integer values so can be used as pixel coords
    centers = [list(imap(int, center)) for center in kmeans.cluster_centers_]
    # Lamda function to remap the y coordiates of the clusters into the image space
    increaseY = lambda points: [points[0] + int((1 - base_size) * img.shape[0]), points[1]]
    # map the centers in terms of the image space
    modifiedCenters = [increaseY(center) for center in centers]

    if modifiedCenters[0][1] < modifiedCenters[1][1]:
        leftx_base = modifiedCenters[0][1]
        rightx_base = modifiedCenters[1][1]
    else:
        leftx_base = modifiedCenters[1][1]
        rightx_base = modifiedCenters[0][1]

    lefty_base = modifiedCenters[0][0]
    righty_base = modifiedCenters[1][0]

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
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    left_d.append(left_fit[3])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    left_fit_[3] = np.mean(left_d[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

    #left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    left_fitx = left_fit_[0]*ploty**3 + left_fit_[1]*ploty**2 + left_fit_[2]*ploty + left_fit_[3]
    #print left_fitx

    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 100, 255] #[255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (leftx_base, lefty_base, rightx_base, righty_base), (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

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
    out_img, base, curves, lanes, ploty = sliding_window(warped_img)

    #cv2.circle(out_img,(base[0],base[1]), 18, (0, 255, 0), -1)
    #cv2.circle(out_img,(base[2],base[3]), 18, (0, 255, 0), -1)

    # Fitted curves as points
    leftLane = np.array([np.transpose(np.vstack([curves[0], ploty]))])
    rightLane = np.array([np.flipud(np.transpose(np.vstack([curves[1], ploty])))])
    points = np.hstack((leftLane, rightLane))
    curves_m = (curves[0]+curves[1])/2
    midLane = np.array([np.transpose(np.vstack([curves_m, ploty]))])

    leftLane_i = leftLane[0].astype(int)
    rightLane_i = rightLane[0].astype(int)
    midLane_i = midLane[0].astype(int)

    cv2.polylines(out_img, [leftLane_i], 0, (0,255,255), thickness=5, lineType=8, shift=0)
    cv2.polylines(out_img, [rightLane_i], 0, (0,255,255), thickness=5, lineType=8, shift=0)
    cv2.polylines(out_img, [midLane_i], 0, (255,0,255), thickness=5, lineType=8, shift=0)

    dst_size =(rwidth, rheight)
    invwarp, Minv = inv_perspective_warp(out_img, dst_size, dst, src)

    midPoints = []
    for i in midLane_i:
      point_wp = np.array([i[0],i[1],1])
      midLane_io = np.matmul(Minv, point_wp) # inverse-M*warp_pt
      midLane_n = np.array([midLane_io[0]/midLane_io[2],midLane_io[1]/midLane_io[2]]) # divide by Z point
      midLane_n = midLane_n.astype(int)
      midPoints.append(midLane_n)

    # Combine the result with the original image
    img_frame[roi_y:height,roi_x:width] = cv2.addWeighted(img_frame[roi_y:height,roi_x:width],
                                                                       1, invwarp, 0.9, 0)
    result = img_frame

    return warped_img, midPoints, out_img, result

def lane_detector():
  publisher = rospy.Publisher('vector_poses', PoseArray, queue_size=10)
  rospy.init_node('lane_detector', anonymous=True)

  rospy.Subscriber("/kinect2_camera/rgb/camera_info", CameraInfo, imagecaminfoCallback)
  rospy.Subscriber("/kinect2_camera/rgb/image_color_rect", Image, imageCallback)

  listener = tf.TransformListener()
  init_transform = geometry_msgs.msg.TransformStamped()

  while not rospy.is_shutdown():

      global img_receive, rgb_img, cam_param_receive, K

      if img_receive==True: #cam_param_receive

          try:
            # wait for the transform to be found
            #listener.waitForTransform("map", "kinect2_rgb_optical_frame", rospy.Time(0),rospy.Duration(5.0))
            (trans,rot) = listener.lookupTransform("map", "kinect2_rgb_optical_frame", rospy.Time(0))

          except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
              continue

          t_c = np.array([[trans[0]], [trans[1]], [trans[2]]])
          R_c = np.quaternion(rot[3], rot[0], rot[1], rot[2]) # Format: (w,x,y,z)

          K_arr = [[K[0], K[1], K[2]],
                   [K[3], K[4], K[5]],
                   [K[6], K[7], K[8]]]

          warped_img, centerLine, curve_fit_img, output = vid_pipeline(rgb_img)

          # Display the resulting frame
          cv2.startWindowThread()
          cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
          cv2.resizeWindow('preview', 800,800)
          cv2.imshow('preview', output)

          # fheight, fwidth = output.shape[:2]
          # warp_img = cv2.resize(warp_img,(int(fwidth),int(fheight)))
          # numpy_horizontal = np.hstack((warp_img, output))
          #
          # cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
          # cv2.resizeWindow('preview', 800,800)
          # cv2.imshow('preview', numpy_horizontal)

          # Total_Points = 5
          # Line_Pts = []
          # # # Used to publish waypoints as pose array so that you can see them in rviz, etc.
          # poses = PoseArray()
          # poses.header.frame_id = "map"
          # poses.header.stamp = rospy.Time.now()
          #
          # for pt in range(Total_Points):
          #
          #  # Line segment points
          #  seg_x = int((centerLine[0][0]*(1-(float(pt)/Total_Points))) + (centerLine[len(centerLine)-1][0]*(float(pt)/Total_Points)))
          #  seg_y = int((centerLine[0][1]*(1-(float(pt)/Total_Points))) + (centerLine[len(centerLine)-1][1]*(float(pt)/Total_Points)))
          #
          #  # Calcuate 3D World Point from 2D Image Point
          #  p_c = np.array([seg_x+roi_x, seg_y+roi_y, 1])
          #  x_c = np.linalg.inv(K_arr).dot(p_c) # Applying Intrinsic Parameters
          #  x_c_norm = LA.norm(x_c, axis=0)
          #  x_c = x_c/x_c_norm # Normalize the vector
          #  x_p = camera2world(x_c, t_c, R_c)
          #
          #  Line_Pts.append([x_p[0],x_p[1]])
          #
          #  position = Point(x_p[0], x_p[1], x_p[2])
          #  orientation = np.quaternion(1,0,0,0)
          #
          #  if pt>0:
          #       yaw = math.atan2(Line_Pts[pt-1][1]-Line_Pts[pt][1],Line_Pts[pt-1][0]-Line_Pts[pt][0])
          #
          #       quaternion_c = tf.transformations.quaternion_from_euler(0, 0, normalizeangle(yaw)) #math.pi
          #       orientation = np.quaternion(quaternion_c[3],quaternion_c[0],quaternion_c[1],quaternion_c[2])
          #
          #  poses.poses.append(Pose(position,orientation))
          #
          # # Publish the vector of poses
          # publisher.publish(poses)

          # Press Q on keyboard to  exit
          if cv2.waitKey(25) & 0xFF == ord('q'):
              break

          # Break the loop
          #else:
              #break

          # Closes all the frames
          #cv2.destroyAllWindows()

          # #Plotting the data
          # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
          # ax1.set_title('Original', fontsize=10)
          # ax1.xaxis.set_visible(False)
          # ax1.yaxis.set_visible(False)
          # ax1.imshow(rgb_img, aspect="auto")
          # ax2.set_title('Filter+Perspective Tform', fontsize=10)
          # ax2.xaxis.set_visible(False)
          # ax2.yaxis.set_visible(False)
          # ax2.imshow(warped_img, aspect="auto")
          #
          # #ax3.plot(curves[0], ploty, color='yellow', linewidth=5)
          # #ax3.plot(curves[1], ploty, color='yellow', linewidth=5)
          # ax3.xaxis.set_visible(False)
          # ax3.yaxis.set_visible(False)
          # ax3.set_title('Sliding window+Curve Fit', fontsize=10)
          # ax3.imshow(curve_fit_img, aspect="auto")
          #
          # ax4.set_title('Overlay Lanes', fontsize=10)
          # ax4.xaxis.set_visible(False)
          # ax4.yaxis.set_visible(False)
          # ax4.imshow(output, aspect="auto")
          #
          # # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
          # plt.show()

          cam_param_receive = False
          rospy.sleep(1)  # sleep for one second

if __name__ == '__main__':
   try:
     lane_detector()
   except rospy.ROSInterruptException:
     pass
