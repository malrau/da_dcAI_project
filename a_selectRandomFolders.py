#######################################
####  FUNCTION TO SELECT A RANDOM  ####
####        SAMPLE OF FOLDERS      ####
#######################################

### IMPORT REQUIRED MODULES ###
import os      # os-like features, as navigating directories
import random  # random

### Selects n random subfolders from a target directory and shows their content
### @param:      targetFolder          [string] - path of folder containing list of subfolders
### @param:                 n             [int] - number of subfolders to sample
### @return: targetSubFolders [list of strings] - sampled subfolders names
### @return:         nFolders [list of strings] - number of folders in sampled subfolders
### @return:           nFiles [list of strings] - number of files in sampled subfolders
def selectRandomFolders(targetFolder, n):
    targetSubFolders = random.sample(os.listdir(targetFolder), n)
    nFolders = list()
    nFiles = list()
    if '.DS_Store' in targetSubFolders:
        targetSubFolders.remove('.DS_Store')
        targetSubFolders.append(random.sample(os.listdir(targetFolder), 1))
    for i in range(0, len(targetSubFolders)):
        for folderPath, folderNames, fileNames in os.walk(targetFolder + '/' + targetSubFolders[i]):
            print('There are', len(folderNames), 'folders and', len(fileNames), 'image files in', folderPath)
            nFolders.append(len(folderNames))
            nFiles.append(len(fileNames))
    return targetSubFolders, nFolders, nFiles
