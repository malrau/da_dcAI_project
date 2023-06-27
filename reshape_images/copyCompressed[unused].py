# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 08:14:09 2023

@author: mzlarosa
"""

import os
import random
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image

# folder containing the original dataset
originFolderTrain = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/train/'
originFolderTest = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/test/'
originFolderValid = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/valid/'

# folder where to put the compressed dataset
destinationFolderTrain = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/compressed/train/'
destinationFolderTest = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/compressed/test/'
destinationFolderValid = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/compressed/valid/'

# subFolders (bird species)
classes = os.listdir(originFolderTrain)

# reproduce subfolders in destination folder
for subFolder in classes:
    if subFolder not in os.listdir(destinationFolderTrain):
        os.makedirs(destinationFolderTrain + subFolder)
    if subFolder not in os.listdir(destinationFolderTest):
        os.makedirs(destinationFolderTest + subFolder)
    if subFolder not in os.listdir(destinationFolderValid):
        os.makedirs(destinationFolderValid + subFolder)

# create compressed images in subfolders
for species in classes:
    for trainFile in os.listdir(originFolderTrain + species):
        trainImage = Image.open(originFolderTrain + species + '/' + trainFile)
        trainImage.save(destinationFolderTrain + species + '/' + trainFile, quality = 50, optimize = True)
    for testFile in os.listdir(originFolderTest + species):
        testImage = Image.open(originFolderTest + species + '/' + testFile)
        testImage.save(destinationFolderTest + species + '/' + testFile, quality = 50, optimize = True)
    for validFile in os.listdir(originFolderValid + species):
        validImage = Image.open(originFolderValid + species + '/' + validFile)
        validImage.save(destinationFolderValid + species + '/' + validFile, quality = 50, optimize = True)

randomImages = list()
for compressedClass in random.sample(os.listdir(destinationFolderTrain), 5):
    for compressedFile in random.sample(os.listdir(destinationFolderTrain + compressedClass), 1):
        compressedImage = mpimg.imread(destinationFolderTrain + compressedClass + '/' + compressedFile)
        randomImages.append(compressedImage)

plt.figure(figsize = (20, 10))
for i in range(0, len(randomImages)):
    plt.subplot(5,5,i+1)
    plt.imshow(randomImages[i])
plt.show()