
# import classes and functions 

import numpy as np     
np.random.seed(0)  #for reproducibility            
 
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten 
from keras.utils import np_utils
import matplotlib.pyplot as plot
 
# Load / Download images if needed 

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Plot Test image in Grey Scale 

plot.subplot(221)
plot.imshow(X_test[0], cmap=plot.get_cmap('gray'))
plot.show()

print(X_test[0].size)
print(X_test[0])  # [28 * 28 Matrix]

# Flatten [28*28] images into single vector [784] for each images using reshape() function of numpy 

X_train = X_train.reshape(60000, 28, 28, 1)     
X_test = X_test.reshape(10000, 28, 28, 1) 
 
X_train = X_train.astype('float32')     
X_test = X_test.astype('float32')     
X_train /= 255    
X_test /= 255
classes = 10  

print(X_test[0].size)
print(X_test[0])  # [1 * 28 * 28 Matrix] 

# It's a multi-class problem, output is 0 to 9. it's a good practice to use "one hot encoding" to class values 

Y_train = np_utils.to_categorical(Y_train, classes)     
Y_test = np_utils.to_categorical(Y_test, classes)

# Set up parameters

input_size = 784
batch_size = 100    
hidden_neurons = 200
epochs = 8

# Build the model
 
model = Sequential() 
model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))  
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))  
                
model.add(Flatten())
  
model.add(Dense(hidden_neurons)) 
model.add(Activation('relu'))      
model.add(Dense(classes)) 
model.add(Activation('softmax'))
      
# compile model 
 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adadelta')

# fit the model 
 
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.1, verbose=1)
 
 # Test 
score = model.evaluate(X_test, Y_test, verbose=1)
print('\n''Test accuracy:', score[1]) 

# save model to create .mlmodel 
model.save('mnist_keras_CNN_model.h5')

# Predict on new data 
# classes = model.predict(x_test)


