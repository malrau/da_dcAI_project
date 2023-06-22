#######################################
### FUNCTIONS TO READ RANDOM IMAGES ###
#######################################

### IMPORT REQUIRED MODULES ###
import os                                   # os-like features, as navigating directories
import random                               # random data generation
from matplotlib import pyplot as plt        # plotting library
from matplotlib import image as mpimg       # dealing with image data


### FUNCTION 1

### Reads one random image from n subfolders ###
### @param:     targetFolder          [string] - path of folder containing list of subfolders
### @param: targetSubFolders [list of strings] - subfolders with image files
### @return:             img [list of strings] - image file names
###
def viewRandomClasses(targetFolder, targetSubFolders):
    n = len(targetSubFolders)

    # empty list where to append image folders paths
    imgFolders = list()
    # empty list where to append one randomly chosen image file name from each folder
    randomImages = list()
    for i in range(0, n):
        imgFolders.append(targetFolder + '/' + targetSubFolders[i])
        randomImages.append(random.sample(os.listdir(imgFolders[i]), 1))

    # read the found images and plot them
    img = list()
    plt.figure(figsize = (25, 15))
    plt.suptitle('\nSample of 15 bird classes', fontsize = 25)
    for i in range(0, n):
        img.append(mpimg.imread(imgFolders[i] + '/' + randomImages[i][0]))
        if n % 5 == 0:
            plt.subplot(n // 5, 5, i + 1)
            plt.title(targetSubFolders[i], fontsize = 15)
        else:
            plt.subplot((n // 5) + 1, 5, i + 1)
            plt.title(targetSubFolders[i], fontsize = 15)
        plt.imshow(img[i])
#       plt.axis('off')
        print('Class:', targetSubFolders[i])
        print(f'Image shape (rows, columns, channels): {img[i].shape}')
        print(f'Image size (number of pixels): {img[i].size}', '\n')
        #print('\n')
    return img


### FUNCTION 2

### Reads n images from a randomly chosen subfolder ###
### @param:     targetFolder          [string] - path of folder containing list of subfolders
### @param: targetSubFolders [list of strings] - subfolders with image files
### @return:             img [list of strings] - image file names
###
def viewRandomClass(targetFolder, targetSubFolders):
    n = len(targetSubFolders)

    # select one random folder from targetSubfolder
    randomSubFolder = random.sample(targetSubFolders, 1)

    # full path of folder from which to draw images
    classFolder = targetFolder + '/' + randomSubFolder[0]

    # list of randomly chosen images within the target folder 
    randomImages = random.sample(os.listdir(classFolder), n)

    # read the found images and plot them
    print(randomSubFolder[0])
    img = list()
    plt.figure(figsize = (25, 15))
    for i in range(0, n):
        if i == 3:
            plt.title(randomSubFolder[0] + '\n', fontsize = 25)
        img.append(mpimg.imread(classFolder + '/' + randomImages[i]))
        if n % 5 == 0:
            plt.subplot(n // 5, 5, i+1)
        else:
            plt.subplot((n // 5) + 1, 5, i + 1)
        plt.imshow(img[i])
#       plt.axis('off')
        print(f'Image shape (rows, columns, channels): {img[i].shape}')
        print(f'Image size (number of pixels): {img[i].size}')
    return img
