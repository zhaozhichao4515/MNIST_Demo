# -*- coding: utf-8 -*-
# import packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data

# 导入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# define  input and output
# input X: 28 x 28 grayscale images,  the first dimension (None) will index the images in the mini-batch
x = tf.placeholder("float",[None, 784])
# correct answers will go here
y_ = tf.placeholder("float", [None, 10])


# weights   W[784,10] 784 = 28 x 28
W = tf.Variable(tf.zeros([784,10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))
# The Network Model: neural network with 1 layer of 10 softmax neurons
y = tf.nn.softmax(tf.matmul(x, W) + b )


## loss function: cross-entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # loss function
# training, learning rate  = 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


sess = tf.Session()
# init
sess.run(tf.initialize_all_variables())
for i in range(1000):
     batch_xs, batch_ys = mnist.train.next_batch(100)    #随机抓取100个批处理数据点训练
     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 训练，将抓取的数据集填充到x 和 y_


# accuracy of trained model, between 0 and 1
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float")) 
print sess.run(accuracy,feed_dict={x: mnist.test.images, y_:mnist.test.labels})