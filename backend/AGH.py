# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:42:32 2019

@author: rioto
"""

import tensorflow as tf
import numpy as np
import os
import cv2
from random import shuffle
from keras.layers import Dense, Dropout, Conv2D, LeakyReLU, BatchNormalization, Flatten, Activation, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
import gc

SHAPE = (256, 256, 3)
NUM_CROPS = 41

class AGH:
        def __init__(self):
            pass
        
        def getData(self, directory):
            oneHotIdx = 0
            imgList = []
            yList = []
            mainDirLength = len(os.listdir(directory))
            
            for dirName, _, pictureList in (os.walk(directory)):
                pictureCount = 0
                
                for pictureName in pictureList:
                    path = dirName + "\\" + pictureName
                    
                    img = cv2.imread(path)
                    
                    if not img is None:
                        pictureCount += 1
                        img = self.preprocess(img)
                        
                        imgList.append(img)
                        yList.append(self.getOneHot(oneHotIdx - 1, mainDirLength))
                    
                print("Done with ", oneHotIdx, " folders!")
                oneHotIdx += 1
            
            data = list(zip(imgList, yList))
            shuffle(data)
            
            imgList, yList = zip(*data)
            
            np.save("data/imgE.npy", imgList)
            np.save("data/predE.npy", yList)
        
        def getOneHot(self, index, numClasses):
            oneHotVector = np.zeros(numClasses)
            oneHotVector[index] = 1
            
            return oneHotVector
            
        def preprocess(self, image):
            image = cv2.resize(image, SHAPE[:2], cv2.INTER_LINEAR)
            
            return image
        
        def loadPickle(self, directory):
            return np.load(directory, allow_pickle = True, mmap_mode = 'r')[:10000]
        
        def loadData(self, directory, dataType):
            img_directory = directory + "/img%s.npy" % dataType 
            pred_directory = directory + "/pred%s.npy" % dataType
            
            data = self.loadPickle(img_directory)
            pred = self.loadPickle(pred_directory)
            
            return data, pred
        
        def createModel(self, model = None):

            optimizer = Adagrad(0.0008)
            kernel = 'he_normal'
            
            if (model == None):
                model = Sequential()
                
                model.add(Conv2D(32, (3, 3), input_shape=SHAPE,
                         kernel_initializer=kernel))
                model.add(LeakyReLU())
                model.add(BatchNormalization())
                model.add(Conv2D(64, (3, 3),
                                 kernel_initializer=kernel))
                model.add(LeakyReLU())
                model.add(MaxPooling2D((2, 2)))
                model.add(Dropout(0.25))
                
                model.add(Conv2D(128, (3, 3),
                                 kernel_initializer=kernel))
                model.add(LeakyReLU())
                model.add(Conv2D(256, (3, 3),
                                 kernel_initializer=kernel))
                model.add(LeakyReLU())
                model.add(MaxPooling2D((2, 2)))
                
                model.add(Conv2D(256, (3, 3),
                                 kernel_initializer=kernel))
                model.add(LeakyReLU())
                model.add(Conv2D(512, (3, 3),
                                 kernel_initializer=kernel))
                model.add(LeakyReLU())
                model.add(MaxPooling2D((2, 2)))
                model.add(Dropout(0.25))
                
                model.add(Dense(1024))
                model.add(Flatten())
                
                model.add(Dense(NUM_CROPS))
                
                model.add(Activation('softmax'))
            else:
                model = load_model(model)
                
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
            print(model.summary())
    
            return model
            
        def train(self, model, name):
            EPOCHS = 10
            BATCH = 16
            
            pictures, predictions = self.loadData("data", "W")
            
            gendata = {
                       'width_shift_range':0.2,
                       'horizontal_flip':True,
                       'vertical_flip':True,
                       'height_shift_range':0.2,
                       'shear_range':0.25,
                       'zoom_range':0.2,
                       'rotation_range':20
                      }
                    
            generator = ImageDataGenerator(**gendata)
            
            generator.fit(pictures)
            
            flow = generator.flow(pictures, predictions, batch_size = BATCH)
            
            model.fit_generator(flow, steps_per_epoch = len(pictures) / BATCH, epochs = EPOCHS)
            
            model.save(name + ".h5")
            
        def evaluate(self, model):
            data, pred = self.loadData("data", "E")
            print(model.evaluate(data, pred))
            
if __name__ == "__main__":
    agh = AGH()
    
    model = "Third.h5"
    model = agh.createModel(model)
    
    for i in range(100):
        gc.collect()
        agh.train(model, "xyz")