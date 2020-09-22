import tensorflow as tf
import numpy as np    
import os
from data_reload import *

# REGULARIZER = 0.01
BATCH_SIZE = 10

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict = {xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict = {xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape, weight_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = weight_name)

def bias_variable(shape, bias_name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = bias_name)

def conv2d(x, W):
    # stride[1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] =1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")  # padding="SAME"用零填充边界，VALID是不填充

def conv2d_2(x, W):
    # stride[1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="VALID")  # padding="SAME"用零填充边界，VALID是不填充

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

def forward(x, keep_prob):
    ## convl layer ##
    x_image = tf.reshape(x, [-1, 324, 80, 1])
    W_conv1 = weight_variable([3, 3, 1, 8], weight_name = "conv1")  # kernel 3*3, channel is 3
    b_conv1 = bias_variable([8], bias_name = "cb1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 322*78*32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 161*39*32

    ## conv2 layer ##
    W_conv2 = weight_variable([3, 3, 8, 16], weight_name = "conv2")  # kernel 3*3, in size 3, out size 5
    b_conv2 = bias_variable([16], bias_name = "cb2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 159*37*5
    h_pool2 = max_pool_2x2(h_conv2)  # output size 79*18*16

    ## funcl layer ##
    W_fc1 = weight_variable([22752, 1024], weight_name = "fc1")
    b_fc1 = bias_variable([1024], bias_name = "fb1")

    # [n_samples,7,7,64]->>[n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 22752])
    h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## func2 layer ##
    W_fc2 = weight_variable([1024, 512], weight_name = "fc2")
    b_fc2 = bias_variable([512], bias_name = "fb2")

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    ## func3 layer ##
    W_fc3 = weight_variable([512, 1], weight_name = "fc3")
    b_fc3 = bias_variable([1], bias_name = "fb3")
    fc3 = tf.matmul(h_fc2_drop, W_fc3)

    y = tf.add(fc3, b_fc3, name = "y")

    return y

def backward():
    datasets, label, test_data, test_label = reload_all_data_R()
    X = tf.placeholder(tf.float32, [None, 324, 80], name = "X")
    Y_ = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
    y = forward(X, keep_prob)
    global_step = tf.Variable(0, trainable=False)
    loss_mse = tf.reduce_mean(tf.square(y-Y_))
    # loss_mse = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(y), reduction_indices=[1]))
    # train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss_mse)
    # train_step=tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse) #其他方法
    train_step=tf.train.AdamOptimizer(0.00001).minimize(loss_mse)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_mse)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 训练模型。
        STEPS = 50000000001
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % 1720
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={X: datasets[start:end], Y_: label[start:end], keep_prob:0.95})
            if i % 200 == 0:
                train_loss = sess.run(loss_mse, feed_dict={X: datasets, Y_: label, keep_prob:1})
                total_loss = sess.run(loss_mse, feed_dict={X: test_data, Y_: test_label, keep_prob:1})
                print("After %d training step(s), loss_mse on train data is %g, loss_mse on val data is %g" % (i, train_loss, total_loss))
            if i % 1000 == 0:
                saver.save(sess, './checkpoint/variable', global_step = i)

def main():
    backward()

if __name__ == '__main__':
    main()
    