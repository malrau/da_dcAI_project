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
The <i>report</i> folder of the project hosts the model results in the form of html and pdf files. Each html-pdf couple is the result of the execution of the <i>dataPrep.py</i> file in Colab. Some of the parameters (batch size and validation generator for model predictions), up to model_v2 and its variations, are defined within the main file (<i>dataPrep.py</i>) and required in the function header. From model_v3 onwards batch size is defined within the model and there is no longer use of validation generator for model predictions. The main changes, however, are made to the implemented sequential model, i.e. the <i>d_sequentialModel.py</i> file. To view changes from one model to the other it is sufficient to open one of the <i>model_results</i> files and navigate to <b><i>section 8) call the sequential model</i></b>. Here, the summary of the model layers and parameters will be helpful in identifying the different model specifications (apart from batch size, which is not shown in the summary).
</p>

### MODELS IMPLEMENTED
<p>
Generally speaking, the model is a sequential one with layers added more or less freely to experiment and test it. The first layer is always the <b><u>Input layer</u></b>, where we define the shape of the data (3D matrices). The core component is the <b><u>Convolutional 2D layer</u></b>, where different kernel sizes and filters are tested (the input size is the same and it is specified in the first layer). Kernel size specifies the dimension of the window on which to apply the convolution. Since it is a 2D convolution, we just need to specify height and width of the window. Filters specifies the number of output filters in the convolution (i.e. the number of feature detectors). Input shape, kernel size and filters determine the number of parameters in the convolutional 2D layer through the formula:<br><br>$` nParameters = ((kernelHeight*kernelWidth*inputShape) + 1)*n_filters `$<br><br>For example, with RGB images as input ($` m*n `$ matrices for three colour channels), if we apply a 2D convolution with kenrel size = (3, 3) and 32 filters we will have a total number of<br><br>$` ((3*3*3) + 1)*32 = 896 `$<br><br>parameters. Usually, after application of a Convolutional 2D layer, I apply a <b><u>MaxPooling 2D layer</u></b> with a pool size of 3: its effect is to downsample each image by taking the maximum value retrieved from a 3x3 pooling window. Pooling, by detecting the main features of an image within a specified window size, is a useful technique to reduce overfitting and speed up training times. Max pooling draws out the most prominent features of an image (max value within window), while average pooling smoothes the image retaining the essence of its features (average value within window). <b><u>Flatten and Dense layers</u></b> are finally added to perform image classification (???). Apart from the model structure in terms of layers, different models can be run on different batch sizes and number of epochs. These parameters are external to the model and determine how many times the model weights are updated and how many times the entire dataset is processed. In particular, the <b>batch size</b> defines the number of data by which a sample of the dataset is composed. By dividing the dimension of the dataset by the batch size we obtain the number of samples processed during each epoch. Usually, for <i>mini-batch gradient descent algorithms</i>, batch sizes of 32, 64 and 128 are used. Theoretically, the only mandatory requirements for the batch size are for it being larger than 0 and smaller or equal to the size of the dataset. A small batch size means that the model is updated more frequently during each epoch, a larger batch size means the model is updated less frequently.
</p>

<br>

<p>
There are essentially three implemented models at the moment in the project results folder.

- ##### model_v1:
this first model is very simple, with a unique Convolutional 2D layer (32 filters and kernel size of 3) and a ReLU (Rectified Linear Unit) activation function, followed by a flatten and a dense image classification layers with a softmax activation function. The batch size is 64 and the model is run for 10 epochs counting an extremely high number of parameters (around 48 millions). The results obtained by the model are quite poor: although the model accuracy is nearly 1 and the loss is very small, model validation is not successful since the accuracy is roughly in the order of $` x*10^{-4} `$ while the loss oscillates between 20 and 30. Testing the model, <b><u>as a consequence???CHECK</u></b>, results in a loss value of 10 and an accuracy of 0.26.
- ##### model_v2:
the second model implementation consists in three Convolutional 2D layers (with ReLU activation functions) followed by a MaxPooling 2D layer each (there is probably a structural error after the first Convolutional 2D layer, since it is followed by the MaxPooling layer rather than the activation function). Image classification layers are a flatten and a dense layer with softmax activation function, like in model_v1. The batch size is 64 and the model is run for 10 epochs. The use of 3 Convolutional 2D layers with MaxPooling 2D greatly reduces the number of parameters which now are around 57 thousands. This results in a harder training process but also in minimising overfitting. The increase in accuracy is softer than before (although it does not achieve more than 0.42 in 10 epochs) and the loss value is slightly higher than 2. Again, validation loss and accuracy are not satisfactory: the first is higher than 30 on average, while the second is even lower than that achieved by model_v1 (it's roughly in the order of $` 6*10^{-5} `$). Also, validation accuracy stays the same from epoch 2 through epoch 10, which doesn't seem normal. Test loss is 9, while accuracy is 0.38.<br>Two variations of the model are presented: the main difference lies in the number of epochs the model is run, which is now 30, while the batch size is the same as model_v2: 64.<br> - <b>model_v2_1</b> obtains a constant and soft increase in model accuracy (which reaches its peak of almost 0.54 in epoch 30) and an equally constant and soft decrease in loss, to a minimum of less than 2 in epoch 30. There seems to be a continuous improvement in model results during training, but this improvement is very slow. Also, model validation is dubious since loss increases to a peak over 43 in epoch 24 (subsequent values do not show a clear trend) while accuracy shows the same unclear constant value of $` 6*10^{-5} `$ we have seen for model_v2 from epoch 2 onwards. Finally, the values of test loss (10) and accuracy (0.45) are similar to those of model_v2.<br> - <b>model_v2_2</b> seems to behave better. [IMPORTANT] It is to be noted that, up until model_v2_1, validation data were obtained by splitting the train dataset: 80 percent was destined to the training process, 20 percent to the validation process. This is <b><u>extremely relevant</u></b> because this kind of splitting is not random but is done by keeping the data as an ordered sequence of elements. My data, though, is a collection of bird classes and the final 20 percent of the data contains classes that are not present in the initial 80 percent of the data. This means that the model was validated by feeding it classes that it was not trained with. While it makes sense that a model is validated with different images than those used during the training, it is harder for it to recognize classes of which it has not seen any images. This explains why validation accuracies were so low. From model_v2_2 onwards, validation data are an entirely different dataset with the same classes of the training dataset but images which are different and smaller in number (5 per class). This explains why the training process is similar to that of model_v2_1 (80 percent of the data are the same, but now training uses also the final 20 percent of the data that were destined to validation before) obtaining a peak accuracy over 0.53 and the lowest loss around 2 both in epoch 30. Model validation, instead, behaves much better: the process shows a behaviour which is coherent with training, although it seems strange that validation accuracy stays higher than the accuracy obtained during training and loss stays lower. Test loss is quite low (1.8) while accuracy is the highest obtained so far (0.56).

- ##### model_v3:
for model_v3 I tried to start the process anew by keeping the important understandings drawn from executing the previous versions of the model and begin again by first implementing a simpler model and subsequently adding layers. Also, I started dropping the use of the validation generator to feed data for prediction. I start using test data for predictions, instead. Hence, in model_v3 there is only one Convolutional 2D layer with ReLU activation function and MaxPooling 2D. This allows to reduce the number of parameters while keeping a certain precision (parameters are around 5 millions). The model has still 64 batch size and it is run for 10 epochs only. Training loss and accuracy reach very good values in epoch 10 (almost 0.06 and 0.99 respectively) while validation loss and accuracy are much less comforting (loss increases to 4.6 in epoch 10, while accuracy increases to 0.45 in epoch 4 but then decreases to almost 0.43 in epoch 10). Test loss and accuracy are coherent with the values obtained during validation (4 and 0.45). I already tested a new model which adds two more Convolutional 2D layers with an increasing number of filters (32-64-128) and two dense layers after the flatten layer. This model has obtained a nice test accuracy of 0.64, with loss of 2.5.
</p>