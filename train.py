import tensorflow as tf
import numpy as np    
import os
from data_reload import *

# REGULARIZER = 0.01
BATCH_SIZE = 10

def get_weight(shape, regularizer, weight_name):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1), name = weight_name)
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape, bias_name):
    b = tf.Variable(tf.zeros(shape), name = bias_name)
    return b

def forward(x, REGULARIZER = 0.01):
    # X = tf.placeholder(tf.float32, shape=(None, 44928))
    # Y = tf.placeholder(tf.float32, shape=(None, 1))
    w1 = get_weight([44928, 1000], REGULARIZER, weight_name = 'w1')
    w2 = get_weight([1000, 100], REGULARIZER, weight_name='w2')
    w3 = get_weight([100, 100], REGULARIZER, weight_name = 'w3')
    w4 = get_weight([100, 1], REGULARIZER, weight_name = 'w4')
    b1 = get_bias([1000], bias_name='b1')
    b2 = get_bias([100], bias_name='b2')
    b3 = get_bias([100], bias_name='b3')
    b4 = get_bias([1], bias_name='b4')
    y1 = tf.nn.tanh(tf.matmul(x, w1) + b1, name = 'y1')
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2, name = 'y2')
    y3 = tf.nn.relu(tf.matmul(y2, w3) + b3, name = 'y3')
    y4 = tf.matmul(y3, w4, name = 'y4')
    y = tf.add(y4, b4, name = 'y')
    return y

def backward(datasets, label, test_data, test_label):
    X = tf.placeholder(tf.float32, [None, 44928], name = "X")
    Y_ = tf.placeholder(tf.float32, [None, 1])
    y = forward(X)
    global_step = tf.Variable(0, trainable=False)
    loss_mse = tf.reduce_mean(tf.square(y-Y_)) 
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
    # train_step=tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse) #其他方法
    # train_step=tf.train.AdamOptimizer(0.001).minimize(loss_mse)  
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 训练模型。
        STEPS = 500001
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % 862
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={X: datasets[start:end], Y_: label[start:end]})
            if i % 200 == 0:
                train_loss = sess.run(loss_mse, feed_dict={X: datasets, Y_: label})
                total_loss = sess.run(loss_mse, feed_dict={X: test_data, Y_: test_label})
                print("After %d training step(s), loss_mse on train data is %g, loss_mse on val data is %g" % (i, train_loss, total_loss))
            if i % 200 == 0:
                saver.save(sess, './checkpoint/variable', global_step = i)
            if(total_loss <= 0.03):
                saver.save(sess, './model/variable')

def main():
    # datasets, label, test_data, test_label = reload_all_data()
    datasets, label, test_data, test_label = Sequential_disruption()
    backward(datasets, label, test_data, test_label)

if __name__ == '__main__':
    main()
    