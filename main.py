import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #import a big dataset
#from tensorflow. 70k images: 55k training, 10k testing, 5k validation
#just a bunch of images of handwritten numbers.
#These images are 28x28 pixels
#All tensors are basically just matrices with a defined shape.

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#downloads/prepares data for our use. Directory is where this happens, turns on
#one-hot encoding: turns target outputs into a vector w/value of 1 or 0. 1's hot
#rest of target outputs are 0. Good for classification.

#Hyperparameters
learning_rate = 0.0001 #how fast to learn.
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
#None means the dimension can be any length.
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
#for it's weights: the number of nodes in the applicable layer.

#Feedforward Calculations
#Using Rectified Linear Unit (ReLU) as the squasher. Output == 0 or input.
#Formula for ReLU: max(0, input) <-- If input is negative, output is 0.
layer_1_output = tf.nn.relu(tf.matmul(network_input,layer_1)+ layer_1_bias)
layer_2_output = tf.nn.relu(tf.matmul(layer_1_output,layer_2) + layer_2_bias)
layer_3_output = tf.nn.relu(tf.matmul(layer_2_output,layer_3) + layer_3_bias)
#Calculates output of every node in layers 1, 2, 3.
#tf.nn.relu is one of ten activation functions offered by TensorFlow. Creates a
#tensor. Takes 1 arg, features. Features must be a certain type of tensor, like
#float32. Here we pass it the product of matmul added to the layer's bias.
#Matmul is a matrix multiplier. Multiplies argA by argB creates a tensor.
#Adds this tensor with layer bias matrix to get total node input.
#For more on matrix operations see links.txt
#That was the summation operator!

network_output_1 = tf.matmul(layer_3_output,output_layer) + output_layer_bias
#This is the same as in the previous lines. Matrix multiplication and added bias
#to get final output of the outputs layer nodes. No squasher yet.
#This output is called a "logit".


network_output_2 = tf.nn.softmax(network_output_1)
#Applying an activation function(squasher) to the logit. Calculates probability
#of each logit sent through. 0-1 probability.
#Softmax is specifically used for classification. Takes 1 arg: tensor of type
#half, float32, or float64.
#Creates final scaled output of the network.

#To train a network we must include the network's prediction, cost function,
#optimization function, and training step.
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=network_output_1, labels=target_output))
#Gets the loss/cost of the the network.
#reduce_mean gets the mean(average) values of a tensor. Also reduces the size or
#dimensions of the array.
#In other words, it calculates the average of elements across tensor dimensions.
#Does this by reducing the shape of the inputted tensor.
#Takes 1 arg: input tensor.
#softmax_cross_entropy_with_logits does two things.
#first applies softmax function to net input of every output node in a network.
#uses network_output_1 -> net input of all output nodes BEFORE squasher applies
#also applies cross_entropy cost function to each of these net outputs. Returns
#final network error/cost/loss. Uses logits of the target output.
#Logits is the first required arg. Logits are the output of all output layer nodes
#without applying a squasher. AKA net input of all output layer nodes. That is why
#we use the pre-softmax output value.
#Labels is second required arg. Target output of every training element. 


train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
#Defines the step/distance the network will take to minimize cost function.
#GradientDescentOptimizer is subclass of Optimizer. Applies gradient descent to the
#weights and biases(how much each affects final output, gives us exact weight combo
#to get to global minimum). Takes only learning rate as arg.
#minimize minimizes the cost/error/loss of a network. Takes only cost/error/loss as
#arg, must be a tensor.


correct_predictions = tf.equal(tf.argmax(network_output_2,1),tf.argmax(target_output,1))
#creates a boolean type tensor object. Checks if any network predictions match labels.
#Answers: which predictions did the network get right?
#tf.equal returns a boolean value based on whether the two args passed to it match.
#Tests if target output and actual output are the same for each output node.
#The two args must be same type tensors. These tensors are defined by argmax.
#tf.argmax "returns the index with the largest value across axes of a tensor"
#Every node output has an index position and value. Index position is it's position
#in an array. The value is a variable, the product of a squasher, the probability an
#input has that that classification.
#Output node with highest value is considered network's prediction.
#argmax finds this highest value and outputs it's index position.
#This essentially finds the largest output node.
#A tensor is a multidimensional array. An axis refers to a row or column of this array.
#Index is the position along this axis.
#argmax Axis arg (in our case, 1): Allows tf to select analyzing a tensor's array by
#row or column. If no axis arg provided, flattens whole array into single row.
#if axis=0: finds max within the columns.
#if axis=1: finds max within the rows.
#Our first use of argmax analyzes every node in network_output_2 and outputs the index
#of one with the highest value.
#The second analyzes every node in target_output and outputs the index of one with the
#highest value.
#tf.equal then checks if these nodes are the same location, aka target/actual output match.



optimizer = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
#finds the average number of correct predictions from correct_predictions.
#correct_predictions is just an array of boolean values. We now take these values,
#convert them to floats, and average them.
#reduce_mean averages the values of tensors.
#cast changes the datatype(dtype) of a tensor. Here it takes booleans of
#correct_predictions, and changes them to float32.
#Two required args: input tensor, dtype to change input into.




#Now we must do the actual training.
#Create a session, loop to train the network and print update statements, and a
#final accuracy statement on completion.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_epochs = 200

    for epoch in range(num_epochs):
        total_cost = 0

        for _ in range(int(mnist.train.num_examples/batch_size)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            t, c = sess.run([train_step,cost_function], feed_dict={network_input:batch_x,target_output:batch_y})
            total_cost += c
        print('Epoch', epoch + 1, 'completed out of', num_epochs, '| loss:', total_cost)
    print('Accuracy:', optimizer.eval({network_input:mnist.test.images, target_output:mnist.test.labels}))


#tf.Session allows us the run the network, aka computational graph.
#sess.run actually invokes the Session. Takes one arg: fetches.
#Fetches can be list, tuple, dict, etc. We use a tensorflow function.
#global_variable_initializer initializes all global variables, delared above, in the graph.
#Global variables are declared with the Variable class.
#An epoch is when an entire training set goes through the network,
#forward and backward.
#num_epochs is the number of times the network will go thru the training set.
#the range(num_epochs) loop loops based on the num_epochs value. ex. 5=5 loops.
#total_cost is obvious. Updates on every epoch completion. Goal is to minimize this #.
#for _ in range(...): defines batches in 1 epoch. Loops over this value.
#Iterates over every batch in an epoch. Batch size, defined above, is 100.
#Batch size is how many training elements to run thru before updating weights.
#mnist.train.num_examples is an instance of DataSet which was imported via input_data.
#num_examples is a variable in said class. Simply total # of training examples.
#batch_x, batch_y = mnist.train... for each training iteration 100 new examples
#are loaded into each variable. x gets training example, y get corresponding target.
#next_batch fills variables with required data, up to the int arg provided.
#t represents training step, c represents cost. This line feeds batches in previous line
#into network_input and target_output.
#train_step and cost_function are both evaluated every batch. Repeats for each epoch.
#t and c hold the temporary step and cost of each batch.
#feed_dict allows to supply data into a tensor. We feed batches of 100 images into
#network_input and target_output. Gives their input and expected output. Allows for
#comparison of target vs actual output.
#network_input and target_output were placeholders defined earlier.
#network_input is fed batch_x, the images (input).
#target_output is fed batch_y, the images repective labels.
#total_cost += c updates total cost by adding the batches current cost.
#Epoch print gives updates about status of which epoch we are on, out of num_epochs.
#Acc print prints the finall network accuracy. Does so by running 10k test images after
#epochs are complete. The decimal given afterward represents a percentage of how
#well the network did. ex. 0.94 = 94% accuracy.
#.eval is fed the dataset's testing images and labels.









