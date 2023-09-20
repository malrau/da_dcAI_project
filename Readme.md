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

<p>
The main file hosting the model is the <i>dataPrep.py</i> file. The file has 10 sections, performing all required operations including data loading, sample plotting, data normalization, model processing and plotting, confusion matrix plotting. Many operations are performed by calling custom functions stored in various .py files. They will be presented under the section description. In detail, we have:

- Preliminary section for importing necessary or useful modules (including files storing my custom functions)
- ##### Section 1:
setting the folders hosting the dataset and showing how many subfolders (bird classes) are there<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;/
- ##### Section 2:
showing how many images are there within a sample of 15 subfolders (bird classes)<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>selectRandomFolders</code>: Selects n random subfolders from a target directory and shows their content
- ##### Section 3:
plotting one random image from the sample chosen in <i>Section 2</i><br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>viewRandomClasses</code>: Reads one random image from n subfolders
- ##### Section 4:
plotting fifteen random images from one of the sampled subfolders<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>viewRandomClass</code>: Reads n images from a randomly chosen subfolder
- ##### Section 5:
loading train, test and validation data and labels<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>selectData</code>: Selects images to be included in the train, test or validation data set
- ##### Section 6:
processing train, test and validation labels into categorical arrays<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>classesToInt</code>: Transforms strings into integers<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>countLabels</code>: Counts distinct integers in array of integer labels
- ##### Section 7:
normalizing train, test and validation data by dividing them by their max value<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;/
- ##### Section 8:
calling and running the sequential model<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>seqModel</code>: includes model definition, printing of summary information on layers' outputs and parameters, model compilation and training, model evaluation and predictions
- ##### Section 9:
plotting model accuracy and loss function<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;/
- ##### Section 10:
plotting the confusion matrix and classification report for a sample of bird classes<br>custom functions:<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>plotCM</code>: Plots the confusion matrix for an image classification model
</p>

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
Porting the project to Colab entailed a unique major challenge, i.e. understanding how to use access the data (bird images) from an online programming setting. The best approach seemed to compress the data set (the train, test and validation folders) and upload it on a <i>Google Drive</i> (the cloud storage solution from Google) folder. To use the data from a <i>Google Drive</i> folder, I needed to import the <code>drive</code> module, which is a dependency of the <code>google.colab</code> module (which is already in memory in a Colab environment) and the <code>sys</code> module. The <code>drive.mount()</code> command mounts the <i>Google Drive</i> account of the user (Colab requires a Google account to run code) where specified, while the <code>sys.path.insert()</code>) command allows to include (<i>insert</i>) the folder where the project is stored in the system folder. In this way it is possible to automatically import dependencies (inserting it at first [0] position makes Python first search that folder for required dependencies). Finally, <code>unzip <i>myArchive.zip</i></code> extracts the content of the archived data set to the Colab environment for use within the notebook. The Colab folders where the data set is stored contain now the target folders that must be fed to the functions used by the project. This means that, relative to the <i>dataPrep.py</i> file, I only had to change the first two sections of the main project file. The project instances that I run in Colab are stored in the <i>colab</i> folder of the project, while the obtained results are stored in the <i>model_results</i> folder (located within the <i>report</i> folder of the project) as html and pdf files.
</p>

<br>

MODEL VERSIONS AND RESULTS
-----------------------------

<p>
The <i>report</i> folder of the project hosts the model results in the form of html and pdf files. Each html-pdf couple is the result of the execution of the <i>dataPrep.py</i> file in Colab. However, the <i>dataPrep.py</i> file does not change from one execution to the other, what really changes is the implemented sequential model, i.e. the <i>d_sequentialModel.py</i> file. To view changes from one model to the other it is sufficient to open one of the <i>model_results</i> files and navigate to <b><i>section 8) call the sequential model</i></b>. Here, the summary of the model layers and parameters will be helpful in identifying the different model specifications..
</p>

### MODELS IMPLEMENTED
<p>
There are essentially three implemented models at the moment in the project results folder:

- ##### model_v1:
this first model is very simple, with a unique Convolutional 2D layer and a ReLU (Rectified Linear Unit) activation function, followed by a flatten and a dense image classification layers with a softmax activation function. The model is run for 10 epochs and counts an extremely high number of parameters (around 48 millions). The results obtained by the model are quite poor: although the model accuracy is nearly 1 and the loss is very small, model validation is not successful since the accuracy is roughly the size of $ x*10^{-4} $ while the loss oscillates between 20 and 30. Testing the model, as a consequence, results in a loss value of 10 and an accuracy of 0.26.
- ##### model_v2:
the second model implementation consists in three Convolutional 2D layers (with ReLU activation functions) followed by a MaxPooling 2D layer each. Image classification layers are a flatten and a dense layer with softmax activation function, like in model_v1. The use of 3 Convolutional 2D layers with MaxPooling 2D greatly reduces the number of parameters which now are around 57 thousands. This results <b><u>(IS THIS TRUE???)</u></b> in a harder training process which obtains a softer increase in accuracy (although not achieving more than 0.42 in 10 epochs) and a loss value slightly higher than 2. Again, validation loss and accuracy are not satisfactory: the first is higher than 30 on average, while the second is even lower than that achieved by model_v1 (it's roughly the size of $ 6*10^{-5} $). Also, validation accuracy stays the same from epoch 2 through epoch 10, which doesn't seem normal. Test loss is 9, while accuracy is 0.38.<br>Two variations of the model are presented: the main difference lies in the number of epochs the model is run, which is now 30. model_v2_1 uses the same batch size as model_v2 <b><u>(WHICH ONE? CHECK!!!)</u></b>, while model v2_2 uses a lower batch size <b><u>(WHICH ONE? CHECK!!!)</u></b>.<br> - model_v2_1 obtains a constant and soft increase in model accuracy (which reaches its peak of almost 0.54 in epoch 30) and an equally constant and soft decrease in loss, to a minimum of less than 2 in epoch 30. There seems to be a continuous improvement in model results during training, but this improvement is very slow. Also, model validation is dubious since loss increases to a peak over 43 in epoch 24 (subsequent values do not show a clear trend) while accuracy shows the same unclear constant value of $ 6*10^{-5} $ we have seen for model_v2 from epoch 2 onwards. Finally, the values of test loss (10) and accuracy (0.45) are similar to those of model_v2.<br> - model_v2_2 seems to behave better: the training process is similar to that of model_v2_1 obtaining a peak accuracy over 0.53 and the lowest loss around 2 both in epoch 30. Model validation seemingly behaves much better, though: the process shows a behaviour which is coherent with training, although it seems strange that validation accuracy stays higher than the accuracy obtained during training and loss stays lower. Test loss is quite low (1.8) while accuracy is the highest obtained so far (0.56).

- ##### model_v3:
for model_v3 I tried to start a new process by first simplifying things and subsequently adding layers. In fact, here there is only one Convolutional 2D layer with ReLU activation function and MaxPooling 2D. This allows to reduce the number of parameters while keeping a certain precision (parameters are around 5 millions). The model is run for 10 epochs only. Training loss and accuracy reach very good values in epoch 10 (almost 0.06 and 0.99 respectively) while validation loss and accuracy are much less comforting (loss increases to 4.6 in epoch 10, while accuracy increases to 0.45 in epoch 4 but then decreases to almost 0.43 in epoch 10). Test loss and accuracy are coherent with the values obtained during validation (4 and 0.45). I already tested a new model which adds two more Convolutional 2D layers with an increasing number of filters (32-64-128) and two dense layers after the flatten layer. This model has obtained a nice test accuracy of 0.64, with loss of 2.5.
</p>