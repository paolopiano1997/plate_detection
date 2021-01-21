from openvino.inference_engine import IENetwork, IECore
import numpy as np
import os
import tensorflow as tf
import cv2
import sys
from tensorflow.keras import backend as K
import string
from timeit import default_timer as timer
import ntpath

model_xml = "saved_model.xml"
model_bin = "saved_model.bin"
ie = IECore()
net = IENetwork(model=model_xml,weights=model_bin)
input_blob = next(iter(net.inputs))
n,c,h,w = net.inputs[input_blob].shape
exec_net = ie.load_network(network=net, device_name="CPU")
input_size=(1,128,64)
img = preprocess(img, input_size=input_size)
img = normalization([img])
img = np.squeeze(img,axis=3)
img = np.expand_dims(img, axis=0)
start = timer()
print("Starting inference...")
res = exec_net.infer(inputs={input_blob: img})
end = timer()
print("End inference time: ", 1000*(end-start))
output_data = res['dense/BiasAdd/Softmax']
print(output_data)
