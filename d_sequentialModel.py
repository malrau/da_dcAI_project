#######################################
####  MACHINE LEARNING MODEL FOR   ####
####     IMAGE CLASSIFICATION      ####
#######################################

### IMPORT REQUIRED MODULES ###
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Activation, Flatten, Dense 

### Defines, compiles, trains and evaluates a sequential model for image classification ###
### @param:     [*]Data   [3D array of integers] - the train, test and validation data
### @param:   [*]Labels   [1D array of integers] - the train, test and validation labels
### @return:    history    [dictionary of lists] - accuracy and loss output from training the model
### @return: prediction      [array of integers] - predicted classification of the model on the test data
###
def seqModel(trainData, trainLabels, testData, testLabels, validationData, validationLabels):
    # Model definition: input and convolutional layers
    model = Sequential()
    model.add(InputLayer(input_shape = (56, 56, 3), name = 'Input')) # input layer
    model.add(Conv2D(filters = 32, kernel_size = 3)) # 2D convolution layer
    model.add(Activation('ReLU', name = 'ReLU'))     # activation function
    model.add(MaxPooling2D(pool_size = (3, 3)))

    # Model definition: image classification layers
    model.add(Flatten())
    model.add(Dense(515, name = 'FC2'))
    model.add(Activation('softmax', name = 'Softmax'))

    # Print a summary of the layers' outputs and model parameters
    model.summary()

    # Compile and train the model (choose loss function, gradient descent...)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Assign accuracy and loss values to history object
    history = model.fit(trainData, trainLabels, epochs = 10, batch_size = 64, validation_data = (validationData, validationLabels))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(testData, testLabels, verbose=0)
    print('\nTest loss:', test_loss, '\nTest accuracy:', test_acc)

    # Predict
    prediction = model.predict(testData)

    return history, prediction
