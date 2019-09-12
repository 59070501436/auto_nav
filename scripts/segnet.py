#!/usr/bin/env python
import keras_segmentation

model = keras_segmentation.models.unet.vgg_segnet(n_classes=50 ,  input_height=360 , input_width=640)

model.train(
    train_images =  "/home/Third_Paper/Frogn_Dataset/images_prepped_train/",
    train_annotations = "/home/Third_Paper/Frogn_Dataset/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_segnet" , epochs=5
)

# out = model.predict_segmentation(
#     inp="dataset1/images_prepped_test/0016E5_07965.png",
#     out_fname="/tmp/out.png"
# )
#
#
# import matplotlib.pyplot as plt
# plt.imshow(out)
