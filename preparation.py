import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib
import requests

from google_drive_downloader import GoogleDriveDownloader as gdd
from matplotlib import pyplot as plt
from PIL import Image

OUTPUT_DIR = 'data/plate_detection'
IMAGE_DIR = os.path.join(OUTPUT_DIR, 'images/ger')
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, 'annotations')
LABELS_CSV = os.path.join(OUTPUT_DIR, 'labels.csv')

gdd.download_file_from_google_drive(file_id='1qYkURFNtmveo4lmEMQfJ1M6HYtt60gNk',
                                    dest_path='data/plate_detection.zip',
                                    unzip=True)
                                    
