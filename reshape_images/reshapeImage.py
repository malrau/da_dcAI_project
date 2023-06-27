# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:57:11 2023

@author: mzlarosa
"""

'''
This file presents a routine for changing the shape of image files
by using the cv2 module, a library for computer vision which contains
useful tools for image processing.
'''

#############################
### IMPORT NEEDED MODULES ###
#############################
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
import numpy as np


#############################
###  MANAGE FILE FOLDERS  ###
#############################
# import files to Google Colab (optional)
'''
from google.colab import files
files.upload()
'''
# set the folder where the files of interest are located
#originFolder = '/content/' # file folder in colab
originFolder = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/Test/'
# try viewing one image from one of the subfolders of the origin folder
classes = os.listdir(originFolder)
images = list()
for species in classes:
    if species == 'SUNBITTERN':
        for img in os.listdir(originFolder + species):
            image = mpimg.imread(originFolder + species + '/' + img)
            plt.imshow(image)

#############################
### CHOOSE IMAGES TO VIEW ###
#############################
# choosing two images to inspect and edit
content = os.listdir(originFolder)[0]
myContent = os.listdir(originFolder + content)[0 : 2]
myContent
# setting up empty lists to fill with the two images
mpimgImages = list()
cv2Images = list()

#############################
###  SHOW IMAGES - MPIMG  ###
#############################
# reading images with imread from matplotlib.Image
plt.figure()
for i in range(0, len(myContent)):
  image = mpimg.imread(originFolder + content + '/' + myContent[i])
  imageType = type(image)
  imageShape = image.shape
  mpimgImages.append(imageType)
  mpimgImages.append(imageShape)
  plt.subplot(1, 2, i + 1)
  plt.title('mpimg')
  plt.imshow(image)
print()
print('Images read with the mpimg module have types: %s and %s, and shapes %s and %s.' % (mpimgImages[0], mpimgImages[2], mpimgImages[1], mpimgImages[3]))

#############################
###   SHOW IMAGES - CV2   ###
#############################
# reading images with imread from cv2
plt.figure()
for i in range(0, len(myContent)):
  imageCV2 = cv2.imread(originFolder + content + '/' + myContent[i])
  imageCV2Type = type(imageCV2)
  imageCV2Shape = imageCV2.shape
  cv2Images.append(imageCV2Type)
  cv2Images.append(imageCV2Shape)
  plt.subplot(1, 2, i + 1)
  plt.title('cv2')
  plt.imshow(imageCV2)
print('Images read with the cv2 module have types: %s and %s, and shapes %s and %s.' % (cv2Images[0], cv2Images[2], cv2Images[1], cv2Images[3]))

#############################
### EDIT ONE IMAGE - CV2  ###
#############################
# choosing one image read with cv2 and show it in original shape (224, 224, 3)
plt.figure()
myCV2Image = cv2.imread(originFolder + content + '/' + myContent[1])
plt.imshow(myCV2Image)
print(type(myCV2Image), myCV2Image.shape)
plt.savefig('/Users/mzlarosa/Downloads/bird.jpg')
# show chosen image in half shape (112, 112, 3)
plt.figure()
myHalfCV2Image = cv2.resize(myCV2Image, (112, 112), interpolation = cv2.INTER_AREA)
plt.imshow(myHalfCV2Image)
print(type(myHalfCV2Image), myHalfCV2Image.shape)
plt.savefig('/Users/mzlarosa/Downloads/halfBird.jpg')
# show chosen image in half shape (112, 112, 3) with different interpolation method
plt.figure()
mySecondHalfCV2Image = cv2.resize(myCV2Image, (112, 112), interpolation = cv2.INTER_NEAREST)
plt.imshow(mySecondHalfCV2Image)
print(type(mySecondHalfCV2Image), mySecondHalfCV2Image.shape)
# show chosen image in a quarter of its shape (112, 112, 3)
plt.figure()
myQuarterCV2Image = cv2.resize(myCV2Image, (56, 56), interpolation = cv2.INTER_AREA)
plt.imshow(myQuarterCV2Image)
print(type(myQuarterCV2Image), myQuarterCV2Image.shape)
plt.savefig('/Users/mzlarosa/Downloads/quarterBird.jpg')

#############################
###  SHOW IMAGE AS ARRAY  ###
#############################
# show the first channel matrix of the original and resized images
print('Original image')
print(myCV2Image[:, :, 0])
print('\nHalf-sized image')
print(myHalfCV2Image[:, :, 0])
print('\nQuarter-sized image')
print(myQuarterCV2Image[:, :, 0])
# show the last point from the first channel matrix of the original and resized images
print('Original image')
print(myCV2Image[223, 223, 0])
print('\nHalf-sized image')
print(myHalfCV2Image[111, 111, 0])
print('\nQuarter-sized image')
print(myQuarterCV2Image[55, 55, 0])

#############################
### SHOW IMAGE IN WINDOWS ###
#############################
# show the original image in a window fitting its shape (224 height, 224 width)
plt.figure(figsize = (2.24, 2.24))
plt.imshow(myCV2Image)
print(type(myCV2Image), myCV2Image.shape)
# show the half-sized image in a window fitting its shape (112 height, 112 width)
plt.figure(figsize = (1.12, 1.12))
plt.imshow(myHalfCV2Image)
print(type(myHalfCV2Image), myHalfCV2Image.shape)
# show the quarter-sized image in a window fitting its shape (56 height, 56 width)
plt.figure(figsize = (0.56, 0.56))
plt.imshow(myQuarterCV2Image)
print(type(myQuarterCV2Image), myQuarterCV2Image.shape)