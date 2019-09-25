#!/usr/bin/env python
import keras_segmentation
from keras_segmentation.predict import model_from_checkpoint_path
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser
import argparse

parser = argparse.ArgumentParser(description="Run evaluation of model on a set of images and annotations.")
parser.add_argument("--model_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/pre-trained_weights/segnet_weights/segnet", help = "Prefix of model filename")
parser.add_argument("--train_images_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/Frogn_Dataset/images_prepped_train/")
parser.add_argument("--train_annotations_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/Frogn_Dataset/annotations_prepped_train/")
parser.add_argument("--inp_dir_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/Frogn_Dataset/images_prepped_test/")
parser.add_argument("--out_dir_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/predicted_outputs_segnet/")
parser.add_argument("--inp_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/Frogn_Dataset/images_prepped_test/frogn_10000.png")
parser.add_argument("--pre_trained", default = "True", type=bool)
parser.add_argument("--predict_multiple_images", default = "False", type=bool)
args = parser.parse_args()

model = keras_segmentation.models.segnet.segnet(n_classes=4,  input_height=320 , input_width=640)
#pre_trained = True
#predict_multiple_images = False

if args.pre_trained:
    model = model_from_checkpoint_path(args.model_path)

else:
    model.train(
        train_images =  args.train_images_path,
        train_annotations = args.train_annotations_path,
        checkpoints_path = args.model_path, epochs=5
    )

if args.predict_multiple_images:
    out = model.predict_multiple(
          inp_dir = args.inp_dir_path,
          checkpoints_path = args.model_path,
          out_dir = args.out_dir_path
     )

else:
    out = model.predict_segmentation(
        inp= args.inp_path,
        checkpoints_path = args.model_path,
        out_fname=os.path.expanduser('~')+"/out.png"
    )

# import matplotlib.pyplot as plt
# plt.imshow(out)
# plt.show()
