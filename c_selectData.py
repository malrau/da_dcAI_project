#######################################
#### FUNCTIONS TO SELECT TRAIN AND ####
#### TEST DATA FOR THE MAIN MODEL  ####
#######################################

### IMPORT REQUIRED MODULES ###
import os                              # os-like features, as navigating directories
from matplotlib import image as mpimg  # dealing with image data


### FUNCTION 1

### Selects images to be included in the train, test or validation data set ###
### @param:  targetFolder          [string] - path of folder containing list of subfolders
### @return:         data  [list of arrays] - image files
### @return:      classes [list of strings] - image files folders (bird classes)
###
def selectData(targetFolder):
    data = list()
    classes = list()
    targetSubFolders = os.listdir(targetFolder) # subfolders with image files
    if '.DS_Store' in targetSubFolders:
        targetSubFolders.remove('.DS_Store')
    for birdClass in targetSubFolders:
        for imageName in os.listdir(targetFolder + '/' + birdClass):
            imagePath = targetFolder + '/' + birdClass + '/' + imageName
            image = mpimg.imread(imagePath)
            if image.shape == (56, 56, 3):
                data.append(image)
                classes.append(birdClass)
            else:
                print('Image', imageName, 'in folder [...]/train/' + birdClass,
                      'has not an homogenous shape. I remove it from the dataset')
    return data, classes


### FUNCTION 2

### Transforms strings into integers (for using tf.keras.utils.to_categorical) ###
### @param:   stringClasses [list of strings] - strings with classes names
### @return:    intClasses [list of integers] - classes names converted in integers
###
def classesToInt(stringClasses):
    intClasses = list()
    for i in range(0, len(stringClasses)):
        if i == 0:
            intClasses.append(0)
        else:
            if stringClasses[i] == stringClasses[i - 1]:
                intClasses.append(intClasses[i - 1])
            else:
                intClasses.append(intClasses[i - 1] + 1)
    return intClasses


### FUNCTION 3

### Counts distinct integers in array of integer labels ###
### @param:  labelsArray  [list of integers] - list of bird classes in dataset
### @return:       count           [integer] - number of bird classes in dataset
###
def countLabels(labelsArray):
    count = 1
    for i in range(0, len(labelsArray)):
        if i > 0 and labelsArray[i] != labelsArray[i - 1]:
            count += 1
    return count