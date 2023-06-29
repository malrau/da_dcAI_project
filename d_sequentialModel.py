#######################################
####  MACHINE LEARNING MODEL FOR   ####
####     IMAGE CLASSIFICATION      ####
#######################################

### IMPORT REQUIRED MODULES ###
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Activation, Flatten, Dense 

def seqModel(trainData, trainLabels, testData, testLabels, batch_size, validation_generator, num_of_test_samples):
    # Model definition: input and convolutional layers
    model = Sequential()
    model.add(InputLayer(input_shape = (56, 56, 3), name = 'Input')) # input layer
    model.add(Conv2D(filters = 16, kernel_size = 3)) # first 2D convolution layer
    model.add(MaxPooling2D(3))
    model.add(Activation('ReLU', name = 'ReLU'))     # activation function
    model.add(Conv2D(filters = 32, kernel_size = 3)) # second 2D convolution layer
    model.add(Activation('ReLU', name = 'ReLU2'))    # activation function
    model.add(MaxPooling2D(3))
    model.add(Conv2D(filters = 64, kernel_size = 3)) # third 2D convolution layer
    model.add(Activation('ReLU', name = 'ReLU3'))    # activation function
    model.add(MaxPooling2D(3))

    # Model definition: image classification layers
    model.add(Flatten())
    model.add(Dense(515, name = 'FC2'))
    model.add(Activation('softmax', name = 'Softmax'))

    # Print a summary of the layers' outputs and model parameters
    model.summary()

    # Compile and train the model (choose loss function, gradient descent...)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Assign accuracy and loss values to history object
    history = model.fit(trainData, trainLabels, epochs = 10, batch_size = batch_size, validation_split = 0.20)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(testData, testLabels, verbose=0)
    print('\nTest loss:', test_loss, '\nTest accuracy:', test_acc)

    # Predict
    prediction = model.predict(validation_generator, num_of_test_samples // (batch_size + 1))

    return history, prediction
