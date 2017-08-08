
import numpy as np
import tensorflow as tf 
import os
from keras.models import model_from_json

sess = tf.Session()


from keras import backend as K 
K.set_session(sess)

# input 

img = tf.placeholder(tf.float32, shape = (None, 784))

# use Keras layers to speed up model definition 

from keras.layers import Dense

# Keras layers can be called Tensor flow tensors 

firstLayer =  Dense(128, activation = 'relu')(img)
secondLayer = Dense(128, activation = 'relu')(firstLayer)

# output layer has 10 units and softmax activation as an activation fn 

predictions = Dense(10,activation = 'softmax')(secondLayer)

# Define Labels/Class value and loss function 

labels = tf.placeholder(tf.float32, shape = (None, 10))
from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels,predictions))


# Let's Train the model with tensor flow optimizer 

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)

optimizer = tf.train.GradientDescentOptimizer(0.5)
model = optimizer.minimize(loss)


# Initialize all variables 
init_operation = tf.global_variables_initializer()
sess.run(init_operation)

# Run the training loops 
with sess.as_default() :
	for i in range(1000): 
		batch = mnist_data.train.next_batch(50)
		print(i)
		#print(mnist_data.train.images[i])
		model.run(feed_dict = { img : batch[0], labels : batch[1] })

# save the model 
saver = tf.train.Saver()
saver.save(sess, 'my_test_model')

# restore model 

# Evaluate model 

from keras.metrics import categorical_accuracy as accuracy 
accuracy_value = accuracy(labels, predictions)
with sess.as_default() :
	print(accuracy_value.eval(feed_dict = { img : mnist_data.test.images, labels : mnist_data.test.labels }))
	print(accuracy_value)






