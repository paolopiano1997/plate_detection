import tensorflow
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.models import load_model

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

BATCH_SIZE = 128
NUM_EPOCHS = 1

# supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta
OPTIMIZER = 'adagrad'

OUTPUT_PATH = 'output/license_recognition'
DOCUMENTATION_PATH = 'documentation'
MODEL_NAME = 'glpr-model'

MODEL_WEIGHTS_PATH = os.path.join(OUTPUT_PATH, OPTIMIZER, MODEL_NAME) + '-weights.h5'
MODEL_PATH = os.path.join(OUTPUT_PATH, OPTIMIZER, MODEL_NAME) + ".h5"
TEST_IMAGES = 'data/license_recognition/test_images'

keras.backend.clear_session()

"""model = load_model("glpr-model-weights.h5",custom_objects={
    'ctc_loss_lambda_func': train_model_keras.CTCLoss
    }, compile=False)
model.compile(loss=train.CTCLoss)

model = load_model("glpr-model-weights.h5",compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite = converter.convert()
open("tflite_model.tflite","wb").write(tflite)
"""

from license_plate_image_augmentor import LicensePlateImageAugmentor
from utils.io import Hdf5DatasetLoader

loader = Hdf5DatasetLoader()
background_images = loader.load(GLP_HDF5, shuffle=True, max_items=10000)
images, labels = loader.load(BACKGRND_HDF5, shuffle=True)
augmentor = LicensePlateImageAugmentor(IMAGE_WIDTH, IMAGE_HEIGHT, background_images)
print("Loaded images " + str(images) + " and labels " + str(labels))
    
class CTCLoss(tf.keras.losses.Loss):

    def __init__(self, input_length, label_length, name='CTCLoss'):
        super().__init__(name=name)
        self.input_length = input_length
        self.label_length = label_length

    def call(self, labels, predictions):
        return tf.keras.backend.ctc_batch_cost(labels, predictions, self.input_length, self.label_length)

#tf.compat.v1.disable_eager_execution()

labels = Input(name='labels', shape=(MAX_TEXT_LEN,), dtype='float32')
input_length = Input(name='input_length', shape=(1,), dtype='int64')
label_length = Input(name='label_length', shape=(1,), dtype='int64')

inputs, outputs = OCR.conv_bgru((IMAGE_WIDTH, IMAGE_HEIGHT, 1), len(LabelCodec.ALPHABET) + 1)


#functional_model = create_functional_model()
#functional_model.save_weights("pretrained_weights.h5")

# In a separate program:
pretrained_model = Model(inputs=[inputs, labels, input_length, label_length], outputs=outputs)

pretrained_model.load_weights("glpr-model-weights.h5")

# Create a new model by extracting layers from the original model:
extracted_layers = pretrained_model.layers[:-1]
print(extracted_layers)
model = keras.Sequential(extracted_layers)
model.summary()


#tf.compat.v1.disable_v2_behavior()
"""
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
        
from tensorflow.compat.v1.keras import backend as K
frozen_graph = freeze_session(K.get_session(), output_names=[str(out.name) for out in model.outputs])
tf.train.write_graph(frozen_graph,".","output.pb",as_text=False)
"""
#model.save("saved_model/license")
