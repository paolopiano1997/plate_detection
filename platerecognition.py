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
import zipfile
import matplotlib
import requests

from google_drive_downloader import GoogleDriveDownloader as gdd
from imutils import paths
from PIL import Image

from config.license_recognition import config

#gdd.download_file_from_google_drive(file_id='1wfOXLXMvmcj-rLsyJXmZ8sIOCo0YlaCE',
#                                   dest_path='data/license_recognition.zip',
#                                   unzip=True)
states_df = pd.read_json("data/license_recognition/german_states.json")
print (states_df)

states_dict = states_df.set_index('DESCR').to_dict()['CM']

counties_df = pd.read_csv("data/license_recognition/KFZ-Deutschland-2017-06-20.csv", delimiter=';')

counties_df['Autokennzeichen'] = counties_df['Autokennzeichen'].str.replace('*','')
counties_df['Bundesland'] = counties_df['Bundesland'].str.split(' ').str[0]

counties_df.insert(1,'State', counties_df['Bundesland'].replace(states_dict))

print ('Number of distinguishing marks: {}'.format(len(counties_df['Autokennzeichen'])))

counties_df.head(30)

class GermanLicensePlateImagesGenerator:
    def __init__(self, counties_df, output_folder):
        self.output_folder = output_folder
        self.counties_df = counties_df
        self.LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ"
        self.DIGITS = "0123456789"
        self.MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        self.YEARS = ['06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']

        random.seed()

    @staticmethod
    def get_image_url(license_number, state, month, year):
        license_number = license_number.replace("-", "%3A").replace("Ä", "%C4").replace("Ö", "%D6").replace("Ü", "%DC")
        return "http://nummernschild.heisnbrg.net/fe/task?action=startTask&kennzeichen={0}&kennzeichenZeile2=&engschrift=false&pixelHoehe=32&breiteInMM=520&breiteInMMFest=true&sonder=FE&dd=01&mm=01&yy=00&kreis={1}&kreisName=&humm={2}&huyy={3}&sonderKreis=LEER&mm1=01&mm2=01&farbe=SCHWARZ&effekt=KEIN&tgaDownload=false".format(
            license_number, state, month, year)

    def generate_license_number(self, county):
        letter_count = random.randint(1, 2)
        letters = "{}".format(random.choice(self.LETTERS)) if letter_count == 1 else "{}{}".format(
            random.choice(self.LETTERS), random.choice(self.LETTERS))

        min = 1 if letter_count == 2 else 1
        digit_count = random.randint(min, max((8 - len(county) - letter_count), 4))
        digits = ""
        for i in range(digit_count):
            digits += random.choice(self.DIGITS)

        return "{}-{}{}".format(county, letters, digits)

    def create_license_plate_image(self, license_number, state, front):
        file_path = os.path.join(self.output_folder, '{0}#{1}.png'.format("F" if front else "R", license_number))
        if os.path.exists(file_path):
            return False

        month = random.choice(self.MONTHS) if front else ''
        year = random.choice(self.YEARS) if front else ''

        create_image_url = GermanLicensePlateImagesGenerator.get_image_url(license_number, state, month, year)
        r = requests.get(create_image_url)
        if r.status_code != 200:
            return False

        id = re.compile('<id>(.*?)</id>', re.DOTALL | re.IGNORECASE).findall(
            r.content.decode("utf-8"))[0]

        status_url = 'http://nummernschild.heisnbrg.net/fe/task?action=status&id=%s' % id
        time.sleep(.200) # wait a short time to not overload the webservice
        r = requests.get(status_url)
        if r.status_code != 200:
            return False

        show_image_url = 'http://nummernschild.heisnbrg.net/fe/task?action=showInPage&id=%s'
        show_image_url = show_image_url % id
        time.sleep(.200) # wait a short time to not overload the webservice
        r = requests.get(show_image_url)
        if r.status_code != 200:
            return False

        # sometimes the web service returns a corrupted image, check the image by getting the size and skip if corrupted
        try:
            numpyarray = np.frombuffer(r.content, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            return Image.fromarray(image)  # don't use cv2.imwrite() because there is a bug with utf-8 encoded filepaths
        except:
            return None

    def generate(self, items):
        for n in range(items):
            while True:
                sample = self.counties_df.sample(1)                
                county = sample.Autokennzeichen.values[0]
                state  = sample.State.values[0]

                license_number = self.generate_license_number(county)
               
                file_name = '{0}#{1}.png'.format("F", license_number)
                file_path = os.path.join(self.output_folder, file_name)
                if os.path.exists(file_path):
                    continue
                
                image = self.create_license_plate_image(license_number, state, True)
                image = image.convert('L')
                if not image is None:
                    image.save(file_path)
                    print("{0:06d} : {1}".format(n+1, file_name))

                time.sleep(.200)

                file_name = '{0}#{1}.png'.format("R", license_number)
                file_path = os.path.join(self.output_folder, file_name)
                if os.path.exists(file_path):
                    continue

                image = self.create_license_plate_image(license_number, state, False)
                image = image.convert('L')
                if not image is None:
                    image.save(file_path)
                    print("{0:06d} : {1}".format(n+1, file_name))

                time.sleep(.200)
                break

        
lpig = GermanLicensePlateImagesGenerator(counties_df, "data/license_recognition/plate_images")


def generate_plate_images():
    sample = lpig.counties_df.sample(1)           
    county = sample.Autokennzeichen.values[0]
    state  = sample.State.values[0]

    print ("County:  {}".format(county))
    print ("State:   {}".format(state))
 
    license_number = lpig.generate_license_number(county)
    print ("License: {}".format(license_number))

    front_image = lpig.create_license_plate_image(license_number, state, True)
    rear_image = lpig.create_license_plate_image(license_number, state, False)
    return np.array(front_image), np.array(rear_image)
    

front_image, rear_image = generate_plate_images()

fig=plt.figure(figsize=(20, 8))

ax = fig.add_subplot(2,2,1)
plt.imshow(cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB))
ax.set_title("Front Plate")
ax =fig.add_subplot(2,2,2)
plt.imshow(cv2.cvtColor(rear_image, cv2.COLOR_BGR2RGB))
ax.set_title("Rear Plate")

ax = fig.add_subplot(2,2,3)
plt.imshow(cv2.cvtColor(front_image, cv2.COLOR_BGR2GRAY), cmap="gray")
ax.set_title("Front Plate")
ax = fig.add_subplot(2,2,4)
plt.imshow(cv2.cvtColor(rear_image, cv2.COLOR_BGR2GRAY), cmap="gray")
ax.set_title("Rear Plate")

plt.show()