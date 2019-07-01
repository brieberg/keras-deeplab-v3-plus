from keras.preprocessing import image
from numpy import random
import numpy as np
import scipy.misc as m
import os
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

def dice_loss(y_true, y_pred):
  print(y_pred)
  print(y_true)

  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  # some implementations don't square y_pred
  denominator = tf.reduce_sum(y_true + tf.square(y_pred))

  return numerator / (denominator + K.epsilon())


def preprocess(img, goal_height=256, goal_width=256, label=False):
    if label:
        img = np.array(m.imread(img, mode="L"), dtype=np.int32)
        img = np.expand_dims(img, 2)
    else:
        img = m.imread(img, mode="RGB")  # [..., [2, 0, 1]]
    #width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    # Crop to Goal Size
    desired_width, desired_height = goal_height, goal_width

    #if width < desired_width:
    #    desired_width = width
    #start_x = np.maximum(0, int((width - desired_width) / 2))

    #img = img.crop((start_x, np.maximum(0, height - desired_height), start_x + desired_width, height))
    img = img.resize((desired_width, desired_height))

    img = image.img_to_array(img) / 255

    return img


def get_validation_data(test, test_labels):
    test_data = [preprocess(i) for i in test]
    test_lab = [preprocess(i, label=True) for i in test_labels]
    return (test_data, test_lab)


def isic_generator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 256, 256, 3))
    batch_labels = np.zeros((batch_size, 256, 256, 2))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = random.choice(len(features), 1)
            batch_features[i] = preprocess(features[index[0]])
            lab = np.squeeze(preprocess(labels[index[0]], label=True))
            batch_labels[i, :, :, 0] = (lab < 1).astype(int)
            batch_labels[i, :, :, 1] = (lab >= 1).astype(int)
            # print(batch_labels[i])
        yield batch_features, batch_labels



def iou(y_true, y_pred):
    """

    :param y_true: (BSx256x256x2) containing labels 0-1
    :param y_pred: (BSx256x256x2) containing probs? should be probs
    :return:
    """
    ytf = K.flatten(y_true)
    ypf = K.flatten(y_pred)

    # Select alternating values, since last dim should have contained class probs
    ypf_0 = ypf[::2]
    ypf_1 = ypf[1::2]
    tf.where()




def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
       from pytorch-semseg.
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
