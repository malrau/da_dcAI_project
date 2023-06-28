#######################################
####      FUNCTION TO PLOT A       ####
####       CONFUSION MATRIX        ####
#######################################

### IMPORT REQUIRED MODULES ###
from matplotlib import pyplot as plt                # plotting library
from sklearn.metrics import ConfusionMatrixDisplay  # Confusion Matrix visualization implementation

### Prints the confusion matrix for an image classification model
### @param:      cm        [2D array] - confusion matrix
### @param: classes [list of strings] - image classes names
def plotCM(cm, classes):
    plt.figure(figsize = (6, 6))
    plt.rcParams['font.size'] = 6
    display_cm = ConfusionMatrixDisplay(cm, display_labels = classes)
    display_cm.plot(cmap='Blues', xticks_rotation = 90)
    plt.xticks(fontsize = 6)
    plt.yticks(fontsize = 6)