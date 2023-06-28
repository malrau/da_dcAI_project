# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:47:09 2023

@author: mzlarosa
"""

import os
import cv2

def reshapeDataset(targetFolder, destinationFolder):
    targetSubFolders = os.listdir(targetFolder)
    for birdClass in targetSubFolders:
        os.makedirs(destinationFolder + '/' + birdClass, exist_ok = True)
        for imageName in os.listdir(targetFolder + '/' + birdClass):
            originalImage = cv2.imread(targetFolder + '/' + birdClass + '/' + imageName)
            newShapes = originalImage.shape[0] // 4
            reshapedImage = cv2.resize(originalImage, (newShapes, newShapes), interpolation = cv2.INTER_AREA)
            print(type(reshapedImage), reshapedImage.shape)
            cv2.imwrite(destinationFolder + '/' + birdClass + '/' + imageName, reshapedImage)

# TRAIN DATA
targetTrain = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/train'
destinationTrain = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/reshaped/train'
reshapeDataset(targetTrain, destinationTrain)

# TEST DATA
targetTest = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/test'
destinationTest = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/reshaped/test'
reshapeDataset(targetTest, destinationTest)

# VALID DATA
targetValid = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/archive/valid'
destinationValid = 'C:/Users/mzlarosa/OneDrive - unime.it/Learning/CdL Informatica/Anno II - Devices and circuits for artificial intelligence/project/dataset/reshaped/valid'
reshapeDataset(targetValid, destinationValid)
