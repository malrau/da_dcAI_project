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
from    matplotlib                import pyplot as plt
from    tensorflow.keras.utils    import to_categorical
from    keras.preprocessing.image import ImageDataGenerator
from    sklearn.metrics           import classification_report, confusion_matrix
from    a_selectRandomFolders     import selectRandomFolders
from    b_viewClasses             import viewRandomClasses, viewRandomClass
from    c_selectData              import selectData, classesToInt, countLabels
from    d_sequentialModel         import seqModel
from    e_plotConfusionMatrix     import plotCM


### 1) set path to the directory of the dataset (this is the targetFolder)
###    and show how many bird classes there are in the path and their names
trainPathWin = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/reshaped/train'
testPathWin = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/reshaped/test'
validPathWin = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/reshaped/valid'
trainPathMac = '/Users/mau/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/reshaped/train'
testPathMac = '/Users/mau/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/reshaped/test'
validPathMac = '/Users/mau/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/reshaped/valid'
classes = os.listdir(trainPathWin)
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
targetClasses = selectRandomFolders(trainPathWin, 15)

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
trainData, trainClasses = selectData(trainPathWin)
testData, testClasses = selectData(testPathWin)
print('The Python lists containing the images are turned into numerical arrays',
      'with 4 dimensions: the first one represents the number of images, while',
      'the other three represent the images\' shape, which has been rendered',
      'homogeneous (56, 56, 3) by removing those not matching the common shape.')
# train data
npTrainData = np.array(trainData)
# test data
npTestData = np.array(testData)
print('Our train data array has', npTrainData.ndim, 'dimensions, and a shape of',
      npTrainData.shape, 'for a total number of elements of', npTrainData.size, '.')
print('Our test data array has', npTestData.ndim, 'dimensions, and a shape of',
      npTestData.shape, 'for a total number of elements of', npTestData.size, '.')

### 6) turn the train and test labels into categorical arrays
# train labels
trainClasses = classesToInt(trainClasses)
npTrainClasses = np.array(trainClasses)
npTrainClasses = np.expand_dims(npTrainClasses, axis = 1) # add dimension to array
npTrainLabels = to_categorical(npTrainClasses)
# test labels
testClasses = classesToInt(testClasses)
npTestClasses = np.array(testClasses)
npTestClasses = np.expand_dims(npTestClasses, axis = 1) # add dimension to array
npTestLabels = to_categorical(npTestClasses)
print('\nThe data labels represent the bird class to which the images belong.',
      'They are first converted into 2D arrays and finally turned into',
      'categorical data: bird classes are substituted by numerical categories.')
trainCount = countLabels(npTrainClasses) # count train labels
testCount = countLabels(npTestClasses) # count test labels
if trainCount == testCount:
    print('There are', npTrainClasses.size, 'categories (it means that each image',
          'belongs to a category) for a set of', trainCount, 'categories.')
else:
    print('ERROR: train labels count and test labels count don\'t match.')

### 6bis) destroy unused objects and free up memory
del trainData, testData, trainClasses, npTrainClasses, testClasses, npTestClasses

### 7) normalize the data
npTrainData = np.array(npTrainData / npTrainData.max(), dtype = np.float16)
npTestData = np.array(npTestData / npTestData.max(), dtype = np.float16)

### 8) call the sequential model
# define variables needed by the model
batchSize = 64                                          # batch size to be used in model.fit
test_datagen = ImageDataGenerator(rescale = 1. / 255)
nTestSamples = npTestData.shape[0]                      # number of images in test folder
validGen = test_datagen.flow_from_directory(testPathWin,
                                            target_size=(56, 56),
                                            batch_size = 64,
                                            class_mode = 'categorical') # generator to be used in model.predict

myModel, Y_pred = seqModel(npTrainData, npTrainLabels, npTestData, npTestLabels, 
                           batchSize, validGen, nTestSamples)

### 9) plot model accuracy and loss function
# accuracy
plt.figure(figsize=(15, 5))
plt.plot(myModel.history['accuracy'], 'r', label='accuracy')
plt.plot(myModel.history['val_accuracy'], 'b', label='val_acc ')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# loss
plt.figure(figsize=(15, 5))
plt.plot(myModel.history['loss'], 'r', label='loss ')
plt.plot(myModel.history['val_loss'], 'b', label='val_loss ')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()

### 10) plot the confution matrix and the classification report for
###     a small sample of classes
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(validGen.classes, y_pred)
thresh = cm.max() / 2.
tick_marks = np.arange(len(classes))
target_names = classes
print('\nConfusion Matrix (small sample)\n')
print(cm[0 : 15, 0 : 15])
plotCM(cm[0 : 15, 0 : 15], classes[0 : 15])
print('\n\nClassification Report (small sample)\n')
class_report = classification_report(validGen.classes, y_pred, target_names = target_names)
print(class_report[0 : 1000], (' ' * 18) + '[...]\n')
