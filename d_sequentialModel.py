#######################################
####  MACHINE LEARNING MODEL FOR   ####
####     IMAGE CLASSIFICATION      ####
#######################################

### IMPORT REQUIRED MODULES ###
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Activation, Flatten, Conv2D

def seqModel(trainData, trainLabels, testData, testLabels):
    # Build/Define the model
    model = Sequential()
    model.add(InputLayer(input_shape = (56, 56, 3), name = 'Input')) # input layer
    model.add(Conv2D(filters = 32, kernel_size = 3)) # 
    model.add(Activation('ReLU', name = 'ReLU'))
    model.add(Flatten())

    # Classify the image
    model.add(Dense(515, name = 'FC2'))
    model.add(Activation('softmax', name = 'Softmax'))

    model.summary() #plot the model

    # Compile and train the model (choose loss function, gradient descent...)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    history = model.fit(trainData, trainLabels, epochs = 10, batch_size = 64, validation_split = 0.20)

    test_loss, test_acc = model.evaluate(testData, testLabels, verbose=0)
    print('\nTest accuracy:', test_acc)
    
    return history