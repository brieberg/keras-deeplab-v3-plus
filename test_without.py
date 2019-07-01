import numpy as np
import os

from utils import *
from model import Deeplabv3
from PIL import Image



# Test without training on isic data

file_list = recursive_glob(
    "data/ISIC2018_Task1-2_Validation_Input/",
    ".jpg"
)
deeplab_model = Deeplabv3(input_shape=(256, 256, 3), classes=2)
print(file_list)
count = 0

print("Loading weigths from disk")
# load weights into new model
deeplab_model.load_weights('./runs/'+str(8761)+'_model.hdf5')
print("Loaded weigths from disk")


for f in file_list:
    trained_image_width=256
    mean_subtraction_value=127.5
    image = np.array(Image.open(f))

    # resize to max dimension of images from training dataset
    w, h, _ = image.shape
    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h),
                                                                            int(ratio * w))))

    # apply normalization for trained dataset images
    resized_image = (resized_image / mean_subtraction_value) - 1.

    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    # make prediction
    res = deeplab_model.predict(np.expand_dims(resized_image, 0))
    labels = np.argmax(res.squeeze(), -1)

    # remove padding and resize back to original image
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    plt.imshow(image)
    plt.imshow(labels, alpha=0.2)
    plt.show()
