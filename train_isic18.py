import numpy as np
import os

from utils import *
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from model import Deeplabv3
import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers.convolutional import Deconvolution2D
from numpy import random
from random import seed, sample, randrange

from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.python.keras.optimizers import Adam


######## Train Model on ISIC
EPOCHS = 10
BS = 2
runnr = randrange(1, 10000)
print("RUNNUMBER", runnr)

deeplab_model = Deeplabv3(input_shape=(256, 256, 3), classes=2)
test_percentage = 0.1

file_list = recursive_glob(
    "/home/bijan/Workspace/Python/keras-deeplab-v3-plus/data/ISIC2018_Task1-2_Training_Input",
    ".jpg"
)

seed(123)
test_indices = sample(range(0, 2594), int(2594 * test_percentage))

test = [sorted(file_list)[k] for k in test_indices]
train = sorted([k for k in file_list if k not in test])

file_list = recursive_glob(
    "/home/bijan/Workspace/Python/keras-deeplab-v3-plus/data/ISIC2018_Task1_Training_GroundTruth/",
    ".png"
)

test_labels = [sorted(file_list)[k] for k in test_indices]
train_labels = sorted(list(set(file_list) - set(test_labels)))


aug = ImageDataGenerator(zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

csv_logger = CSVLogger('./runs/'+str(runnr)+'_log.csv', append=True, separator=';')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
checkpointer = ModelCheckpoint(filepath='./runs/'+str(runnr)+'_model.hdf5',
                               verbose=1,
                               save_weights_only=True,
                               save_best_only=True)


deeplab_model.compile(optimizer=Adam(lr=0.0001), loss=keras.losses.binary_crossentropy, metrics=['binary_accuracy'])

H = deeplab_model.fit_generator(isic_generator(train, train_labels, batch_size=BS),
	                            validation_data=isic_generator(test, test_labels, batch_size=BS),
                                validation_steps=len(test_labels),
                                steps_per_epoch=len(train) // BS,
	                            epochs=EPOCHS,
                                max_queue_size=3,
                                callbacks=[checkpointer, csv_logger, reduce_lr])

model_json = deeplab_model.to_json()
with open("./runs/"+str(runnr)+"_model.json", "w") as json_file:
    json_file.write(model_json)
print(H.history)
