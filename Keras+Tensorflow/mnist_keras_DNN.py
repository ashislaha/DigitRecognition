
# import classes and functions 

from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plot
 
# Load / Download images if needed 

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Plot image in Grey Scale 

print(X_test[0].size)
print(X_test[0])  # [28 * 28 Matrix]

plot.subplot(221)
plot.imshow(X_test[0], cmap=plot.get_cmap('gray'))
plot.show()
 
# Flatten [28*28] images into single vector [784] for each images using reshape() function of numpy 

X_train = X_train.reshape(60000, 784)     
X_test = X_test.reshape(10000, 784)
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
hidden_neurons = 400    
epochs = 30
 
# Build the model
model = Sequential()     
model.add(Dense(hidden_neurons, input_dim=input_size)) 
model.add(Activation('relu'))     
model.add(Dense(classes, input_dim=hidden_neurons)) 
model.add(Activation('softmax'))

# compile model 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adadelta')

# fit the model 

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
 
# Test 
score = model.evaluate(X_test, Y_test, verbose=1)
print('\n''Test accuracy:', score[1]) 

# save model to create .mlmodel 
model.save('mnist_keras_DNN_model.h5')



