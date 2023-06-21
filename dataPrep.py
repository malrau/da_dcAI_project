# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:45:36 2023

@author: mzlarosa
"""

##########################################
### import necessary or useful modules ###
##########################################
import os
import numpy as np
from    matplotlib              import pyplot as plt
from    matplotlib              import image as mpimg
from    tensorflow.keras.utils  import to_categorical
from    a_selectRandomFolders   import selectRandomFolders
from    b_viewClasses           import viewRandomClasses, viewRandomClass
from    c_selectData            import selectData, classesToInt, countLabels


### 1) set path to the directory of the dataset (this is the targetFolder)
###    and show how many bird classes there are in the path and their names
trainPathWin = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/train'
testPathWin = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/test'
trainPath = '/Users/mau/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/train'
testPath = '/Users/mau/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/test'
validPath = '/Users/mau/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/valid'
classes = os.listdir(trainPathWin) # trainPath
if '.DS_Store' in classes:
    classes.remove('.DS_Store')
nOfClasses = len(classes)
print('\nThere are', nOfClasses, 'classes in the dataset.\n' +
      '\nHere is a list of the first 50 classes:')
print(classes[0 : 50], end = '')
print('[...]')

### 2) select a random sample of n (15) subfolders from the targetFolder
###    and show their content (subfolders represent bird classes)
###    modules: os, random, selectRandomFolders
print('\nThe number of available pictures varies with the class.')
print('We select a sample of 15 classes and show how many images',
      'are available for each of them:')
targetClasses = selectRandomFolders(trainPathWin, 5)

### 3) plot one random image from each bird class
###    modules: b_viewClasses
print('\nThen we draw a random picture from each class and show their shape and size.',
      'A picture shape shows tipically three dimensions.',
      'The first two dimensions build a 2D matrix of n rows by m columns.',
      'The number of rows represents the image height in pixels,',
      'while the number of columns represents the image width in pixels.',
      'So each matrix coordinate point represents a single pixel intensity value.',
      'The third dimension refers to the number of color planes (or channels).',
      'There is one plane (2D matrix) for each RGB color,',
      'so the value of the third dimension is usually 3.',
      'A picture size shows its total number of pixels, which results by',
      'multiplying the dimensions of the 2D matrices by themselves and by',
      'the number of matrices (color planes).')
randomClasses = viewRandomClasses(trainPathWin, targetClasses[0])
#plt.show()
plt.savefig('../imgs/randomClasses.png')

### 4) plot 15 random images from one of the classes
###    modules: b_viewClasses
print('\nFinally, we print 15 random images from one of the previously',
      'chosen classes and show their shape and size as before defined.')
randomClass = viewRandomClass(trainPathWin, targetClasses[0])
#plt.show()
plt.savefig('../imgs/randomClass.png')

print('\nWe conclude that, although there is a varying number of images',
      'for each bird class, in our sample of 15 classes there is a minimum',
      'of', min(targetClasses[2]), 'images, and a maximum of',
      max(targetClasses[2]), 'images.\n')

### 5) load the train and test data and labels into memory
train = selectData(trainPathWin)
test = selectData(testPathWin)
print('The Python lists containing the images are turned into numerical arrays',
      'with 4 dimensions: the first one represents the number of images, while',
      'the other three represent the images\' shape, which has been rendered',
      'homogeneous (224, 224, 3) by removing those not matching the common shape.')
# train data
trainData = train[0]
npTrainData = np.array(trainData)
# test data
testData = test[0]
npTestData = np.array(testData)
print('Our train data array has', npTrainData.ndim, 'dimensions, and a shape of',
      npTrainData.shape, 'for a total number of elements of', npTrainData.size, '.')
print('Our test data array has', npTestData.ndim, 'dimensions, and a shape of',
      npTestData.shape, 'for a total number of elements of', npTestData.size, '.')

### 6) turn the train and test labels into categorical arrays
# train labels
trainClasses = classesToInt(train[1])
npTrainClasses = np.array(trainClasses)
npTrainClasses = np.expand_dims(npTrainClasses, axis = 1) # add dimension to array
trainLabels = to_categorical(trainClasses)
# test labels
testClasses = classesToInt(test[1])
npTestClasses = np.array(testClasses)
npTestClasses = np.expand_dims(npTestClasses, axis = 1) # add dimension to array
npTestLabels = to_categorical(npTestClasses)
print('\nThe data labels represent the bird class to which the images belong.',
      'They are first converted into 2D arrays and finally turned into',
      'categorical data: bird classes are substituted by numerical categories.')
trainCount = countLabels(trainClasses) # count train labels
testCount = countLabels(npTestClasses) # count test labels
if trainCount == testCount:
    print('There are', npTestClasses.size, 'categories (it means that each image',
          'belongs to a category) for a set of', testCount, 'categories.')
    else:
        print('ERROR: train labels count and test labels count don\'t match.')

### 7) normalize the data
npTrainData = npTrainData / npTrainData.max()
npTestData = npTestData / npTestData.max()
