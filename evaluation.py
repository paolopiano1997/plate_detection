
import skimage
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import random
import re
import time
import cv2
import requests
import pandas as pd
import progressbar
import tarfile
import tensorflow as tf
import zipfile
import matplotlib
import requests

from google_drive_downloader import GoogleDriveDownloader as gdd
from distutils.version import StrictVersion
from imutils import paths
from PIL import Image
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from utils.nn.conv import OCR
from label_codec import LabelCodec
from train_helper import TrainHelper
from config.license_recognition import config
# datasets
GLP_HDF5 = 'data/license_recognition/background.h5'
BACKGRND_HDF5 = 'data/license_recognition/glp.h5'

# image size <=> network input size
IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT

# training parameter
DOWNSAMPLE_FACTOR = config.DOWNSAMPLE_FACTOR
MAX_TEXT_LEN = config.MAX_TEXT_LEN

BATCH_SIZE = 64
NUM_EPOCHS = 1000

# supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta
OPTIMIZER = 'adagrad'

OUTPUT_PATH = 'output/license_recognition'
DOCUMENTATION_PATH = 'documentation'
MODEL_NAME = 'glpr-model'

MODEL_WEIGHTS_PATH = os.path.join(OUTPUT_PATH, OPTIMIZER, MODEL_NAME) + '-weights.h5'
MODEL_PATH = os.path.join(OUTPUT_PATH, OPTIMIZER, MODEL_NAME) + ".h5"
TEST_IMAGES = 'data/license_recognition/test_images'

def show_image(image):
    plt.axis("off")
    plt.title(label)
    plt.imshow(image, cmap='gray')
    plt.show()
	
from tensorflow.keras.models import load_model

tf.compat.v1.disable_eager_execution()
model = load_model("saved_model/license", compile=False)

from PIL import Image
from utils.preprocessing import AspectAwarePreprocessor

preprocessor = AspectAwarePreprocessor(IMAGE_WIDTH, IMAGE_HEIGHT)

from label_codec import LabelCodec
from matplotlib import gridspec

img_filename = random.choice(os.listdir(TEST_IMAGES))
img_filepath = os.path.join(TEST_IMAGES, img_filename)
label = img_filename.split(".")[0].split("#")[0]

image = Image.open(img_filepath) 

# original image
show_image(image)

# predict
image = preprocessor.preprocess(image)
image = image.astype(np.float32) / 255.

image = np.expand_dims(image.T, axis=-1)

import time
start_time = time.time()
predictions = model.predict(np.asarray([image]))
print("Total time prediction: {}".format(time.time() - start_time))
pred_number = LabelCodec.decode_prediction(predictions[0])


fig = plt.figure(figsize=(15, 10))
outer = gridspec.GridSpec(1, 2, wspace=.5, hspace=0.1)
ax1 = plt.Subplot(fig, outer[0])
fig.add_subplot(ax1)
print('Predicted: %9s\nTrue:      %9s\n=> %s' % (pred_number, label, pred_number == label))
image = image[:, :, 0].T
ax1.set_title('True: {}\nPred: {}'.format(label, pred_number), loc='left')
ax1.imshow(image, cmap='gray')
ax1.set_xticks([])
ax1.set_yticks([])

ax2 = plt.Subplot(fig, outer[1])
fig.add_subplot(ax2)
ax2.set_title('Activations')
ax2.imshow(predictions[0].T, cmap='binary', interpolation='nearest')
ax2.set_yticks(list(range(len(LabelCodec.ALPHABET) + 1)))
ax2.set_yticklabels(LabelCodec.ALPHABET)  # + ['blank'])
ax2.grid(False)
for h in np.arange(-0.5, len(LabelCodec.ALPHABET) + 1 + 0.5, 1):
    ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)
	
