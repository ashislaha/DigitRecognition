
# import classes 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
#Load data set, and split it if necessary 
mnist = input_data.read_data_sets("MNIST_data/")
 
# we create a holder, a container to place the computation activities in tensorflow , identifying format and tensor's r/c, null means any kind

VISIBLE_NODES = 784
HIDDEN_NODES = 400
x = tf.placeholder("float", shape=[None, VISIBLE_NODES])
y = tf.placeholder("float", shape=[None, 10])
 
# We identify weights and biases with tensor shape, start with 0
weights = tf.Variable(tf.random_normal((VISIBLE_NODES, HIDDEN_NODES), 
    mean=0.0, stddev=1. / VISIBLE_NODES))
hidden_bias = tf.Variable(tf.zeros([HIDDEN_NODES]))
visible_bias = tf.Variable(tf.zeros([VISIBLE_NODES]))
 
#set up the sigmoid model and multiply x and W with matmul function, building the hidden layer and reconstruction layer

hidden_activation = tf.nn.sigmoid(tf.matmul(x, weights) + hidden_bias)
visible_reconstruction = tf.nn.sigmoid(tf.matmul(hidden_activation, tf.transpose(weights)) 
    + visible_bias)
final_hidden_activation = tf.nn.sigmoid(tf.matmul(visible_reconstruction, weights) 
    + hidden_bias)
 
# This process can be understood as being two phases of learning positive and negative or, more poetically, waking and sleeping

positive_phase = tf.matmul(tf.transpose(x), hidden_activation)
negative_phase = tf.matmul(tf.transpose(visible_reconstruction), final_hidden_activation)
LEARNING_RATE = 0.01
weight_update = weights.assign_add(LEARNING_RATE *
    (positive_phase - negative_phase))
visible_bias_update = visible_bias.assign_add(LEARNING_RATE *
    tf.reduce_mean(x - visible_reconstruction, 0))
hidden_bias_update = hidden_bias.assign_add(LEARNING_RATE *
    tf.reduce_mean(hidden_activation - final_hidden_activation, 0))


# Now we create the operations for scaling the hidden and visible biases, with loss function feedback 


train_op = tf.group(weight_update, visible_bias_update, hidden_bias_update)
loss_op = tf.reduce_sum(tf.square(x - visible_reconstruction))
 
# We start the session

session = tf.Session()
session.run(tf.global_variables_initializer())
current_epochs = 0
 
# Run the session

for i in range(20):
    total_loss = 0
    while mnist.train.epochs_completed == current_epochs:
        batch_inputs, batch_labels = mnist.train.next_batch(100)
        _, reconstruction_loss = session.run([train_op, loss_op], feed_dict={input_placeholder: batch_inputs})
        total_loss += reconstruction_loss
 
    print("epochs %s loss %s" % (current_epochs, reconstruction_loss))
    current_epochs = mnist.train.epochs_completed



