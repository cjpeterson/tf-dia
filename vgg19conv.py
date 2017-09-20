'''
Implementation of the convolutional portion  of a VGG-19 network (up to
layer 5-1, inclusive). This network was presented in a paper by K. Simonyan and
A. Zisserman, cited below. The model, and the database containing their
weights, are licensed under the Creative Commons Attribution License, which is
accessible at
https://creativecommons.org/licenses/by/4.0/

 Very Deep Convolutional Networks for Large-Scale Image Recognition
 K. Simonyan, A. Zisserman
 arXiv:1409.1556
'''
import tensorflow as tf

class Vgg19Conv:
    
    #BGR
    VGG_MEAN = [103.939, 116.779, 123.68]
    
    def __init__(self, model):
        self.model = model
    
    def get_block(self, input, num):
        if (num == 1):
            conv1_1 = self.conv_layer(input, "conv1_1")
            return conv1_1
        elif (num == 2):
            conv1_2 = self.conv_layer(input, "conv1_2")
            pool1 = self.max_pool(conv1_2, "pool1")
            conv2_1 = self.conv_layer(pool1, "conv2_1")
            return conv2_1
        elif (num == 3):
            conv2_2 = self.conv_layer(input, "conv2_2")
            pool2 = self.max_pool(conv2_2, "pool2")
            conv3_1 = self.conv_layer(pool2, "conv3_1")
            return conv3_1
        elif (num == 4):
            conv3_2 = self.conv_layer(input, "conv3_2")
            conv3_3 = self.conv_layer(conv3_2, "conv3_3")
            conv3_4 = self.conv_layer(conv3_3, "conv3_4")
            pool3 = self.max_pool(conv3_4, "pool3")
            conv4_1 = self.conv_layer(pool3, "conv4_1")
            return conv4_1
        elif (num == 5):
            conv4_2 = self.conv_layer(input, "conv4_2")
            conv4_3 = self.conv_layer(conv4_2, "conv4_3")
            conv4_4 = self.conv_layer(conv4_3, "conv4_4")
            pool4 = self.max_pool(conv4_4, "pool4")
            conv5_1 = self.conv_layer(pool4, "conv5_1")
            return conv5_1
        
        return None
    
    def conv_layer(self, input, name):
        with tf.variable_scope(name):
            filt = tf.Variable(self.model[name][0], name="filter")
            
            conv = tf.nn.conv2d(input, filt, [1, 1, 1, 1], padding='SAME')
            
            conv_biases = tf.Variable(self.model[name][1], name="biases")
            bias = tf.nn.bias_add(conv, conv_biases)
            
            relu = tf.nn.relu(bias)
            return relu
    
    def max_pool(self, value, name):
        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)
