-- 540438 - Maurizio La Rosa --
===============================

<p>
This repository hosts the project for the exam of the course <b>Devices and circuits for artificial intelligence</b> from the Data Analysis degree of the University of Messina. The project consists in building a machine learning model for image classification.

The dataset to be used is hosted at kaggle, at the link https://www.kaggle.com/datasets/gpiosenka/100-bird-species and currently contains images for 525 bird species to be classified by the model. I downloaded the dataset on April, 17th, 2023, and that version contains 515 bird species.

It is useful to note that images in the dataset should have all the exact same shape (224, 224, 3), while I found that all images of the 'PLUSH CRESTED JAY' species and one image from the 'LOGGERHEAD SHRIKE' species have variable shapes. Hence, in my code, I check for images' shapes and remove images that don't match the common shape. This is important because imported images have the shape of 3D Numpy (np, when imported) arrays and I need to transform the list of images into a 4D Numpy array. The function np.array() can do it automatically when fed a list of 3D Numpy arrays, but images must have all the same shape.
</p>

<br>

MAIN STRUCTURE
--------------

The main file hosting the model is the <i>dataPrep.py</i> file. The file has 10 sections, performing all required operations including data loading, sample plotting, data normalization, model processing and plotting, confusion matrix plotting. Many operations are performed by calling custom functions stored in various .py files. They will be presented under the section description. In detail, we have:

- Preliminary section for importing necessary or useful modules (including files storing my custom functions)
- Section 1: setting the folders hosting the dataset and showing how many subfolders (bird classes) are there<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;/
- Section 2: showing how many images are there within a sample of 15 subfolders (bird classes)<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>selectRandomFolders</code>: Selects n random subfolders from a target directory and shows their content
- Section 3: plotting one random image from the sample chosen in <i>Section 2</i><br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>viewRandomClasses</code>: Reads one random image from n subfolders
- Section 4: plotting fifteen random images from one of the sampled subfolders<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>viewRandomClass</code>: Reads n images from a randomly chosen subfolder
- Section 5: loading train, test and validation data and labels<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>selectData</code>: Selects images to be included in the train, test or validation data set
- Section 6: processing train, test and validation labels into categorical arrays<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>classesToInt</code>: Transforms strings into integers<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>countLabels</code>: Counts distinct integers in array of integer labels
- Section 7: normalizing train, test and validation data by dividing fortheir max value<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;/
- Section 8: running the sequential model<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>seqModel</code>: includes model definition, printing of summary information on layers' outputs and parameters, model compilation and training, model evaluation and predictions
- Section 9: plotting model accuracy and loss function<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;/
- Section 10: plotting the confusion matrix and classification report for a sample of bird classes<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>plotCM</code>: Plots the confusion matrix for an image classification model

<br>

TACKLING PERFORMANCE PROBLEMS
-----------------------------

### RAM SHORTAGE
<p>
The structure presented above should run the model from start to end if classes are stored on a local machine. In particular, each class should be a subfolder within the train, test and validation folders. However, I run into performance problems which led me to implement various strategies.<br>First of all I had a memory shortage due to the dimension and number of image arrays. Images are 3D matrices of dimension 224*224*3 and there are hundreds of images for each class and there are more than 500 classes: this means thet, at some point during data normalization, the process was aborted due to lack of memory where to store the original arrays and the processed arrays. I tried to overcome this problem by reshaping the images into 3D matrices of dimension 56*56*3. 
</p>

#### Reshaping images
<p>
The <i>reshape_images</i> folder of the project hosts a couple of files: the <i>reshapeImageData.py</i> file hosts the <code>reshapedataset</code> function, which reduces square images by 1/4 their number of pixels (with contextual application on each of the three main (train, test and validation) folders of the data set). The <i>reshapeImage.py</i> file is a sort of reshaping guide with examples. I also implemented this procedure on a Jupyter notebook and created a small report as a html and a pdf file (these are located under the <i>report</i> folder of the project).
</p>

### EXECUTION TIME
<p>
A second problem lies in the time requested to run the model, which depends on the machine, but approximates one hour on a Core i7 with 16 GB RAM (machine at the office). So, in order to be run in Google Colab, which has plenty of resources (including hardware acceleration) to host machine learning models, I also ported the main code (the <i>dataPrep.py</i> file) to Jupyter notebook, while I kept the custom functions on their respective .py files imported from the main file.
</p>

#### Porting the project to Google Colab
<p>
Porting the project to Colab entailed a unique major challenge, i.e. understanding how to use the data set within an online programming setting. The best approach seemed to compress the data set (the three train, test and validation folders) and upload it on a Google Drive (the Google Cloud storage solution) folder. To use the data from a Google Drive folder, I needed to import the <code>drive</code> module ( a dependency of the <code>google.colab</code> module already in memory in a Colab environment and the <code>sys</code> module. The <code>drive.mount()</code> command mounts the Google Drive account of the user (Colab requires a Google account to run code) where specified, while the <code>sys.path.insert()</code>) command allows to include (<i>insert</i>) the folder where the project is stored in the system folder from which it is possible to automatically import dependencies (inserting it at first [0] position makes Python first search that folder for required dependencies). Finally, <code>unzip <i>myArchive.zip</i></code> extracts the content of the archived data set to the Colab environment for use within the notebook. The Colab folders where the data set is stored contain now the target folders that must be fed to the functions used by the project. This means that, relative to the <i>dataPrep.py</i> file, I only had to change two sections of the main project file. The code attempts run in Colab are stored in the <i>colab</i> folder of the project, while the obtained results are stored in the <i>model_results</i> folder of the <i>report</i> folder of the project as html and pdf files.
</p>