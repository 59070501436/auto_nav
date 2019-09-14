#!/usr/bin/env python
import numpy as np
import cv2
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser
import os.path as osp
import glob

input_dir = expanduser("~/Third_Paper/data_dataset_voc/SegmentationClass/")
output_dir = expanduser("~/Third_Paper/Frogn_Dataset/annotations_prepped_train/")

for label_file in glob.glob(osp.join(input_dir, '*.npy')):
    print('Generating dataset from:', label_file)
    with open(label_file) as f:
        base = osp.splitext(osp.basename(label_file))[0]
        masked_image = np.load(label_file)

        cv2.imwrite(output_dir+base+'.png', masked_image)
