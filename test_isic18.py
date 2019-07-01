from keras.models import model_from_json
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from model import Deeplabv3
from utils import *
import os

runnr = 8761

model = Deeplabv3(input_shape=(256, 256, 3),
                  classes=1,
                  weights=None)

#
# trained_image_width=512
# mean_subtraction_value=127.5
# image = np.array(Image.open('data/ISIC2018_Task1-2_Validation_Input/ISIC_0012255.jpg'))
#
# # resize to max dimension of images from training dataset
# w, h, _ = image.shape
# ratio = float(trained_image_width) / np.max([w, h])
# resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
#
# # apply normalization for trained dataset images
# resized_image = (resized_image / mean_subtraction_value) - 1.
#
# # pad array to square image to match training images
# pad_x = int(trained_image_width - resized_image.shape[0])
# pad_y = int(trained_image_width - resized_image.shape[1])
# resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
#
# # make prediction
# deeplab_model = Deeplabv3(input_shape=(512, 512, 3), classes=2)
# res = deeplab_model.predict(np.expand_dims(resized_image, 0))
# labels = np.argmax(res.squeeze(), -1)
#
# # remove padding and resize back to original image
# if pad_x > 0:
#     labels = labels[:-pad_x]
# if pad_y > 0:
#     labels = labels[:, :-pad_y]
# labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
# plt.imshow(image)
# plt.imshow(labels, alpha=0.2)

print("Loading weigths from disk")
# load weights into new model
model.load_weights('./runs/'+str(runnr)+'_model.hdf5')
print("Loaded weigths from disk")


file_list = recursive_glob("data/ISIC2018_Task1-2_Validation_Input/", ".jpg")

count = 0


def postprocess_label(res,
                      h=256,
                      w=256):
    labels = np.argmax(res.squeeze(), -1)
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    return labels


for f in file_list:
    img = preprocess(f)
    plt.imshow(img)
    expand_img = np.expand_dims(img, 0)
    print(expand_img.shape)
    lbl = model.predict(expand_img)
    print(lbl.shape)
    print(np.max(lbl))
    plt.imshow(postprocess_label(lbl),  alpha=0.2)
    plt.show()
    count += 1

