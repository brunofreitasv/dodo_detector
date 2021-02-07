#!/usr/bin/env python
# coding: utf-8

import numpy as np
from os import listdir, makedirs, remove, walk
from PIL import Image
from tqdm import tqdm
from os.path import isdir, exists, isfile
from dodo_detector.detection import TFObjectDetector
import argparse
import shutil
from detection_params import *

class Detection():

    def __init__(self):
        self.__model = MODEL_DIR
        self.__saved_model = SAVED_MODEL
        self.__labelmap = LABEL_MAP_FILENAME
        self.__imagedir = IMAGES_DIR
        self.__confidence = CONFIDENCE

        self.__savedir = OUTPUT_DIR
        self.__savebboxinfo = SAVE_BBOX_TO_TXT_FILE

    def execute(self):
        detector = TFObjectDetector(
            self.__model + '/' + self.__saved_model,
            self.__model + '/' + self.__labelmap,
            confidence=self.__confidence
            )

        ims = [self.__imagedir + '/' + im for im in listdir(self.__imagedir)]
        ims.sort()

        if exists(self.__savedir + '/marked_images') and isdir(self.__savedir + '/marked_images'):
            shutil.rmtree(self.__savedir + '/marked_images')
        makedirs(self.__savedir + '/marked_images')

    
        if self.__savebboxinfo:
            if exists(self.__savedir + '/detection_results') and isdir(self.__savedir + '/detection_results'):
                shutil.rmtree(self.__savedir + '/detection_results')
            makedirs(self.__savedir + '/detection_results')

        for im in tqdm(ims):
            self.__generate_output_files(detector, im)
            
    def __generate_output_files(self, detector, im):
        img = np.array(Image.open(im))
        marked_image, objects = detector.from_image(img)

        if self.__savebboxinfo:            
            result_file = self.__savedir + '/detection_results/' + (im.split('/')[-1]).replace('.jpg','.txt')
            with open(result_file, "a") as myfile:
                for clas, detections in objects.iteritems():
                    for bbox in detections:
                        line = clas
                        line += ' ' 
                        line += str(bbox['confidence'])
                        line += ' '

                        cord = [str(v) for v in bbox['box']]

                        line += ' '.join([cord[1], cord[0], cord[3], cord[2]])
                        line += '\n'
                        myfile.writelines(line)
        
        result_file = self.__savedir + '/marked_images/'+ (im.split('/')[-1]).replace('.jpg','_marked.jpg')
        Image.fromarray(marked_image).save(result_file)

if __name__ == '__main__':
    Detection().execute()
