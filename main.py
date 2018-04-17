import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #import a big dataset
#from tensorflow. 70k images: 55k training, 10k testing, 5k validation
#just a bunch of images of handwritten numbers.
#These images are 28x28 pixels

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#downloads/prepares data for our use. Directory is where this happens, turns on
#one-hot encoding: turns target outputs into a vector w/value of 1 or 0. 1's hot
#rest of target outputs are 0. Good for classification.

#Hyperparameters
learning_rate = 0.0001
batch_size = 100 #how many training elements to run thru, before an update.
update_step = 10 #how often to print update of training process. In "epochs"
#1 epoch = an entire training set has gone thru/back a network... ours is 55k

#Network Parameters
layer_1_nodes = 500
layer_2_nodes = 500
layer_3_nodes = 500#3 hidden layers, 500 nodes each
output_nodes = 10#one for each possible output (0-9)

#Placeholders - empty tensors, fed data as network operates.
#Args - Data type(required), (Both default to None) -> shape, name
#feedforward usually has just 2. For input layer, and for target output layer
network_input = tf.placeholder(tf.float32, [None,784])
target_output = tf.placeholder(tf.float32, [None, output_nodes])
#tf.float32 - a datatype. 32 vs 64 bit - lower resolution but uses less memory
#32 bit FLOATING point number.
#[None, 784] - defines shape of input layer. [length, width]
#Length represents total number of imgs in a batch. Don't know, defined later.
#Width represents width of the input. It's actually total "area" of it.
#28x28 = 784. 784 input nodes, because each input has 784 pixels in the img.
#Could have been defined as a network parameter.

#Tweakable Parameters (by network) - weights and biases, quantity and weights.
layer_1 = tf.Variable(tf.random_normal([784,layer_1_nodes]))
layer_1_bias = tf.Variable(tf.random_normal([layer_1_nodes]))

layer_2 = tf.Variable(tf.random_normal([layer_1_nodes,layer_2_nodes]))
layer_2_bias = tf.Variable(tf.random_normal([layer_2_nodes]))

layer_3 = tf.Variable(tf.random_normal([layer_2_nodes,layer_3_nodes]))
layer_3_bias = tf.Variable(tf.random_normal([layer_3_nodes]))

output_layer = tf.Variable(tf.random_normal([layer_3_nodes, output_nodes]))
output_layer_bias = tf.Variable(tf.random_normal([output_nodes]))

#Variable is a class that creates new tensor to hold/update parameters.
#In this case, they hold weights. One required arg - initial value. many options
#Initial value defines datatype/shape
#random_normal creates random distribution of numbers. Defines initial value of
#tf.Variable, which itself defines datatype and shape.
#This means all weights in defined shape are initialized as random.
#The args 784,layer_1_nodes (both int) are the number of connections between two
#layers in a network, thus defining total number of weights: 784x500.
#Shape for layer_1: 784x500 means there are 392000 weights between input/layer1
#Bias's do not connect to any previous layer. That is why there is only one arg
#for it's weights: the number of nodes in applicable layer.

#Feedforward Calculations
#Using Rectified Linear Unit (ReLU) as the squasher. Output == 0 or input.
layer_1_output = tf.nn.relu(tf.matmul(network_input,layer_1)+ layer_1_bias)
layer_2_output = tf.nn.relu(tf.matmul(layer_1_output,layer_2) + layer_2_bias)
layer_3_output = tf.nn.relu(tf.matmul(layer_2_output,layer_3) + layer_3_bias)



network_output_1 = tf.matmul(layer_3_output,output_layer) + output_layer_bias



network_output_2 = tf.nn.softmax(network_output_1)




