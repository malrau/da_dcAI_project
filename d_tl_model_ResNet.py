#######################################
####  MACHINE LEARNING MODEL FOR   ####
####     IMAGE CLASSIFICATION      ####
#######################################

### IMPORT REQUIRED MODULES ###
from tensorflow.keras.models       import Sequential
from tensorflow.keras.layers       import Activation, Dense
from tensorflow.keras.applications import ResNet50

def seqModel(trainData, trainLabels, testData, testLabels, validationData, validationLabels):
    # Model definition: application of ResNet pre-trained model
    model = Sequential()
    model.add(ResNet50(input_shape = (56, 56, 3),
                       include_top = False,
                       pooling = 'avg',
                       weights = 'imagenet'))
    
    # Model definition: image classification layers
    model.add(Dense(515, name = 'FC'))
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
