#!/usr/bin/env python
import os
import cv2
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser
import glob
from PIL import Image

roi_x = 120
roi_y = 208

for d in range(5151): #5151
        home = expanduser("~/Third_Paper/Datasets/Frogn_Fields/Frogn_005/frogn_1%04d.png"%d)
        print(home)

        img = cv2.imread(home)
        # cv2.imshow('preview', img)
        # k = cv2.waitKey(1) & 0xFF
        # if k == 27:
        #    break

        # Getting ROI
        iheight, iwidth = img.shape[:2]
        Roi = img[0:iheight-roi_y,0:iwidth-roi_x]

        # Store the Cropped Images
        Filelocation = expanduser("~/Third_Paper/Datasets/Frogn_Fields/Frogn_005_cropped/frogn_1%04d.png"%d)
        cv2.imwrite(Filelocation, Roi)
