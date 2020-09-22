import tensorflow as tf
import numpy as np
import os
from data_reload import *

# REGULARIZER = 0.01
BATCH_SIZE = 10


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape, weight_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=weight_name)


def bias_variable(shape, bias_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=bias_name)


def conv2d(x, W):
    # stride[1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] =1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")  # padding="SAME"用零填充边界，VALID是不填充


def conv2d_2(x, W):
    # stride[1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="VALID")  # padding="SAME"用零填充边界，VALID是不填充


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

def forward(x, keep_prob):
    x1 = tf.reshape(x[0], [-1, 40, 40, 1])
    x2 = tf.reshape(x[1], [-1, 40, 40, 1])
    x3 = tf.reshape(x[2], [-1, 40, 40, 1])
    x4 = tf.reshape(x[3], [-1, 40, 40, 1])
    x5 = tf.reshape(x[4], [-1, 40, 40, 1])
    x6 = tf.reshape(x[5], [-1, 40, 40, 1])
    x7 = tf.reshape(x[6], [-1, 40, 40, 1])
    x8 = tf.reshape(x[7], [-1, 40, 40, 1])
    x9 = tf.reshape(x[8], [-1, 40, 40, 1])
    x10 = tf.reshape(x[9], [-1, 40, 40, 1])
    x11 = tf.reshape(x[10], [-1, 40, 40, 1])
    x12 = tf.reshape(x[1], [-1, 40, 40, 1])
    x13 = tf.reshape(x[12], [-1, 40, 40, 1])
    x14 = tf.reshape(x[13], [-1, 40, 40, 1])
    x15 = tf.reshape(x[14], [-1, 40, 40, 1])
    x16 = tf.reshape(x[15], [-1, 40, 40, 1])

    ## convl layer ##
    W_conv1_1 = weight_variable([3, 3, 1, 8], weight_name="conv1_1")  # kernel 3*3, channel is 3
    b_conv1_1 = bias_variable([8], bias_name="cb1_1")
    h_conv1_1 = tf.nn.relu(conv2d(x1, W_conv1_1) + b_conv1_1)  # output size 38*38*8
    h_pool1_1 = avg_pool_2x2(h_conv1_1)  # output size 19*19*8

    W_conv1_2 = weight_variable([3, 3, 1, 8], weight_name="conv1_2")  # kernel 3*3, channel is 3
    b_conv1_2 = bias_variable([8], bias_name="cb1_2")
    h_conv1_2 = tf.nn.relu(conv2d(x2, W_conv1_2) + b_conv1_2)  # output size 38*38*8
    h_pool1_2 = avg_pool_2x2(h_conv1_2)  # output size 19*19*8

    W_conv1_3 = weight_variable([3, 3, 1, 8], weight_name="conv1_3")  # kernel 3*3, channel is 3
    b_conv1_3 = bias_variable([8], bias_name="cb1_3")
    h_conv1_3 = tf.nn.relu(conv2d(x3, W_conv1_3) + b_conv1_3)  # output size 38*38*8
    h_pool1_3 = avg_pool_2x2(h_conv1_3)  # output size 19*19*8

    W_conv1_4 = weight_variable([3, 3, 1, 8], weight_name="conv1_4")  # kernel 3*3, channel is 3
    b_conv1_4 = bias_variable([8], bias_name="cb1_4")
    h_conv1_4 = tf.nn.relu(conv2d(x4, W_conv1_4) + b_conv1_4)  # output size 38*38*8
    h_pool1_4 = avg_pool_2x2(h_conv1_4)  # output size 19*19*8

    W_conv1_5 = weight_variable([3, 3, 1, 8], weight_name="conv1_5")  # kernel 3*3, channel is 3
    b_conv1_5 = bias_variable([8], bias_name="cb1_5")
    h_conv1_5 = tf.nn.relu(conv2d(x5, W_conv1_5) + b_conv1_5)  # output size 38*38*8
    h_pool1_5 = avg_pool_2x2(h_conv1_5)  # output size 19*19*8

    W_conv1_6 = weight_variable([3, 3, 1, 8], weight_name="conv1_6")  # kernel 3*3, channel is 3
    b_conv1_6 = bias_variable([8], bias_name="cb1_6")
    h_conv1_6 = tf.nn.relu(conv2d(x6, W_conv1_6) + b_conv1_6)  # output size 38*38*8
    h_pool1_6 = avg_pool_2x2(h_conv1_6)  # output size 19*19*8

    W_conv1_7 = weight_variable([3, 3, 1, 8], weight_name="conv1_7")  # kernel 3*3, channel is 3
    b_conv1_7 = bias_variable([8], bias_name="cb1_7")
    h_conv1_7 = tf.nn.relu(conv2d(x7, W_conv1_7) + b_conv1_7)  # output size 38*38*8
    h_pool1_7 = avg_pool_2x2(h_conv1_7)  # output size 19*19*8

    W_conv1_8 = weight_variable([3, 3, 1, 8], weight_name="conv1_8")  # kernel 3*3, channel is 3
    b_conv1_8 = bias_variable([8], bias_name="cb1_8")
    h_conv1_8 = tf.nn.relu(conv2d(x8, W_conv1_8) + b_conv1_8)  # output size 38*38*8
    h_pool1_8 = avg_pool_2x2(h_conv1_8)  # output size 19*19*8

    W_conv1_9 = weight_variable([3, 3, 1, 8], weight_name="conv1_9")  # kernel 3*3, channel is 3
    b_conv1_9 = bias_variable([8], bias_name="cb1_9")
    h_conv1_9 = tf.nn.relu(conv2d(x9, W_conv1_9) + b_conv1_9)  # output size 38*38*8
    h_pool1_9 = avg_pool_2x2(h_conv1_9)  # output size 19*19*8

    W_conv1_10 = weight_variable([3, 3, 1, 8], weight_name="conv1_10")  # kernel 3*3, channel is 3
    b_conv1_10 = bias_variable([8], bias_name="cb1_10")
    h_conv1_10 = tf.nn.relu(conv2d(x10, W_conv1_10) + b_conv1_10)  # output size 38*38*8
    h_pool1_10 = avg_pool_2x2(h_conv1_10)  # output size 19*19*8

    W_conv1_11 = weight_variable([3, 3, 1, 8], weight_name="conv1_11")  # kernel 3*3, channel is 3
    b_conv1_11 = bias_variable([8], bias_name="cb1_11")
    h_conv1_11 = tf.nn.relu(conv2d(x11, W_conv1_11) + b_conv1_11)  # output size 38*38*8
    h_pool1_11 = avg_pool_2x2(h_conv1_11)  # output size 19*19*8

    W_conv1_12 = weight_variable([3, 3, 1, 8], weight_name="conv1_12")  # kernel 3*3, channel is 3
    b_conv1_12 = bias_variable([8], bias_name="cb1_12")
    h_conv1_12 = tf.nn.relu(conv2d(x12, W_conv1_12) + b_conv1_12)  # output size 38*38*8
    h_pool1_12 = avg_pool_2x2(h_conv1_12)  # output size 19*19*8

    W_conv1_13 = weight_variable([3, 3, 1, 8], weight_name="conv1_13")  # kernel 3*3, channel is 3
    b_conv1_13 = bias_variable([8], bias_name="cb1_13")
    h_conv1_13 = tf.nn.relu(conv2d(x13, W_conv1_13) + b_conv1_13)  # output size 38*38*8
    h_pool1_13 = avg_pool_2x2(h_conv1_13)  # output size 19*19*8

    W_conv1_14 = weight_variable([3, 3, 1, 8], weight_name="conv1_14")  # kernel 3*3, channel is 3
    b_conv1_14 = bias_variable([8], bias_name="cb1_14")
    h_conv1_14 = tf.nn.relu(conv2d(x14, W_conv1_14) + b_conv1_14)  # output size 38*38*8
    h_pool1_14 = avg_pool_2x2(h_conv1_14)  # output size 19*19*8

    W_conv1_15 = weight_variable([3, 3, 1, 8], weight_name="conv1_15")  # kernel 3*3, channel is 3
    b_conv1_15 = bias_variable([8], bias_name="cb1_15")
    h_conv1_15 = tf.nn.relu(conv2d(x15, W_conv1_15) + b_conv1_15)  # output size 38*38*8
    h_pool1_15 = avg_pool_2x2(h_conv1_15)  # output size 19*19*8

    W_conv1_16 = weight_variable([3, 3, 1, 8], weight_name="conv1_16")  # kernel 3*3, channel is 3
    b_conv1_16 = bias_variable([8], bias_name="cb1_16")
    h_conv1_16 = tf.nn.relu(conv2d(x16, W_conv1_16) + b_conv1_16)  # output size 38*38*8
    h_pool1_16 = avg_pool_2x2(h_conv1_16)  # output size 19*19*8

########################################################################################################################
########################################################################################################################
########################################################################################################################

    ## conv2 layer ##
    W_conv2_1 = weight_variable([3, 3, 8, 16], weight_name="conv2_1")  # kernel 3*3, in size 3, out size 5
    b_conv2_1 = bias_variable([16], bias_name="cb2_1")
    h_conv2_1 = tf.nn.relu(conv2d(h_pool1_1, W_conv2_1) + b_conv2_1)  # output size 17*17*16
    h_pool2_1 = avg_pool_2x2(h_conv2_1)  # output size 8*8*16

    W_conv2_2 = weight_variable([3, 3, 8, 16], weight_name="conv2_2")  # kernel 3*3, in size 3, out size 5
    b_conv2_2 = bias_variable([16], bias_name="cb2_2")
    h_conv2_2 = tf.nn.relu(conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)  # output size 17*17*16
    h_pool2_2 = avg_pool_2x2(h_conv2_2)  # output size 8*8*16

    W_conv2_3 = weight_variable([3, 3, 8, 16], weight_name="conv2_3")  # kernel 3*3, in size 3, out size 5
    b_conv2_3 = bias_variable([16], bias_name="cb2_3")
    h_conv2_3 = tf.nn.relu(conv2d(h_pool1_3, W_conv2_3) + b_conv2_3)  # output size 17*17*16
    h_pool2_3 = avg_pool_2x2(h_conv2_3)  # output size 8*8*16

    W_conv2_4 = weight_variable([3, 3, 8, 16], weight_name="conv2_4")  # kernel 3*3, in size 3, out size 5
    b_conv2_4 = bias_variable([16], bias_name="cb2_4")
    h_conv2_4 = tf.nn.relu(conv2d(h_pool1_4, W_conv2_4) + b_conv2_4)  # output size 17*17*16
    h_pool2_4 = avg_pool_2x2(h_conv2_4)  # output size 8*8*16

    W_conv2_5 = weight_variable([3, 3, 8, 16], weight_name="conv2_5")  # kernel 3*3, in size 3, out size 5
    b_conv2_5 = bias_variable([16], bias_name="cb2_5")
    h_conv2_5 = tf.nn.relu(conv2d(h_pool1_5, W_conv2_5) + b_conv2_5)  # output size 17*17*16
    h_pool2_5 = avg_pool_2x2(h_conv2_5)  # output size 8*8*16

    W_conv2_6 = weight_variable([3, 3, 8, 16], weight_name="conv2_6")  # kernel 3*3, in size 3, out size 5
    b_conv2_6 = bias_variable([16], bias_name="cb2_6")
    h_conv2_6 = tf.nn.relu(conv2d(h_pool1_6, W_conv2_6) + b_conv2_6)  # output size 17*17*16
    h_pool2_6 = avg_pool_2x2(h_conv2_6)  # output size 8*8*16

    W_conv2_7 = weight_variable([3, 3, 8, 16], weight_name="conv2_7")  # kernel 3*3, in size 3, out size 5
    b_conv2_7 = bias_variable([16], bias_name="cb2_7")
    h_conv2_7 = tf.nn.relu(conv2d(h_pool1_7, W_conv2_7) + b_conv2_7)  # output size 17*17*16
    h_pool2_7 = avg_pool_2x2(h_conv2_7)  # output size 8*8*16

    W_conv2_8 = weight_variable([3, 3, 8, 16], weight_name="conv2_8")  # kernel 3*3, in size 3, out size 5
    b_conv2_8 = bias_variable([16], bias_name="cb2_8")
    h_conv2_8 = tf.nn.relu(conv2d(h_pool1_8, W_conv2_8) + b_conv2_8)  # output size 17*17*16
    h_pool2_8= avg_pool_2x2(h_conv2_8)  # output size 8*8*16

    W_conv2_9 = weight_variable([3, 3, 8, 16], weight_name="conv2_9")  # kernel 3*3, in size 3, out size 5
    b_conv2_9 = bias_variable([16], bias_name="cb2_9")
    h_conv2_9 = tf.nn.relu(conv2d(h_pool1_9, W_conv2_9) + b_conv2_9)  # output size 17*17*16
    h_pool2_9 = avg_pool_2x2(h_conv2_9)  # output size 8*8*16

    W_conv2_10 = weight_variable([3, 3, 8, 16], weight_name="conv2_10")  # kernel 3*3, in size 3, out size 5
    b_conv2_10 = bias_variable([16], bias_name="cb2_10")
    h_conv2_10 = tf.nn.relu(conv2d(h_pool1_10, W_conv2_10) + b_conv2_10)  # output size 17*17*16
    h_pool2_10 = avg_pool_2x2(h_conv2_10)  # output size 8*8*16

    W_conv2_11 = weight_variable([3, 3, 8, 16], weight_name="conv2_11")  # kernel 3*3, in size 3, out size 5
    b_conv2_11 = bias_variable([16], bias_name="cb2_11")
    h_conv2_11 = tf.nn.relu(conv2d(h_pool1_11, W_conv2_11) + b_conv2_11)  # output size 17*17*16
    h_pool2_11 = avg_pool_2x2(h_conv2_11)  # output size 8*8*16

    W_conv2_12 = weight_variable([3, 3, 8, 16], weight_name="conv2_12")  # kernel 3*3, in size 3, out size 5
    b_conv2_12 = bias_variable([16], bias_name="cb2_12")
    h_conv2_12 = tf.nn.relu(conv2d(h_pool1_12, W_conv2_12) + b_conv2_12)  # output size 17*17*16
    h_pool2_12 = avg_pool_2x2(h_conv2_12)  # output size 8*8*16

    W_conv2_13 = weight_variable([3, 3, 8, 16], weight_name="conv2_13")  # kernel 3*3, in size 3, out size 5
    b_conv2_13 = bias_variable([16], bias_name="cb2_13")
    h_conv2_13 = tf.nn.relu(conv2d(h_pool1_13, W_conv2_13) + b_conv2_13)  # output size 17*17*16
    h_pool2_13 = avg_pool_2x2(h_conv2_13)  # output size 8*8*16

    W_conv2_14 = weight_variable([3, 3, 8, 16], weight_name="conv2_14")  # kernel 3*3, in size 3, out size 5
    b_conv2_14 = bias_variable([16], bias_name="cb2_14")
    h_conv2_14 = tf.nn.relu(conv2d(h_pool1_14, W_conv2_14) + b_conv2_14)  # output size 17*17*16
    h_pool2_14 = avg_pool_2x2(h_conv2_14)  # output size 8*8*16

    W_conv2_15 = weight_variable([3, 3, 8, 16], weight_name="conv2_15")  # kernel 3*3, in size 3, out size 5
    b_conv2_15 = bias_variable([16], bias_name="cb2_15")
    h_conv2_15 = tf.nn.relu(conv2d(h_pool1_15, W_conv2_15) + b_conv2_15)  # output size 17*17*16
    h_pool2_15 = avg_pool_2x2(h_conv2_15)  # output size 8*8*16

    W_conv2_16 = weight_variable([3, 3, 8, 16], weight_name="conv2_16")  # kernel 3*3, in size 3, out size 5
    b_conv2_16 = bias_variable([16], bias_name="cb2_16")
    h_conv2_16 = tf.nn.relu(conv2d(h_pool1_16, W_conv2_16) + b_conv2_16)  # output size 17*17*16
    h_pool2_16 = avg_pool_2x2(h_conv2_16)  # output size 8*8*16

    ## funcl layer ##
    W_fc1 = weight_variable([16384, 1024], weight_name="fc1")
    b_fc1 = bias_variable([1024], bias_name="fb1")

    # [n_samples,7,7,64]->>[n_samples, 7*7*64]
    h_pool2_flat_1 = tf.reshape(h_pool2_1, [-1, 8*8*16])
    h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, 8*8*16])
    h_pool2_flat_3 = tf.reshape(h_pool2_3, [-1, 8*8*16])
    h_pool2_flat_4 = tf.reshape(h_pool2_4, [-1, 8*8*16])
    h_pool2_flat_5 = tf.reshape(h_pool2_5, [-1, 8*8*16])
    h_pool2_flat_6 = tf.reshape(h_pool2_6, [-1, 8*8*16])
    h_pool2_flat_7 = tf.reshape(h_pool2_7, [-1, 8*8*16])
    h_pool2_flat_8 = tf.reshape(h_pool2_8, [-1, 8*8*16])
    h_pool2_flat_9 = tf.reshape(h_pool2_9, [-1, 8*8*16])
    h_pool2_flat_10 = tf.reshape(h_pool2_10, [-1, 8*8*16])
    h_pool2_flat_11 = tf.reshape(h_pool2_11, [-1, 8*8*16])
    h_pool2_flat_12 = tf.reshape(h_pool2_12, [-1, 8*8*16])
    h_pool2_flat_13 = tf.reshape(h_pool2_13, [-1, 8*8*16])
    h_pool2_flat_14 = tf.reshape(h_pool2_14, [-1, 8*8*16])
    h_pool2_flat_15 = tf.reshape(h_pool2_15, [-1, 8*8*16])
    h_pool2_flat_16 = tf.reshape(h_pool2_16, [-1, 8*8*16])
    h_pool2_flat = tf.concat([h_pool2_flat_1, h_pool2_flat_2, h_pool2_flat_3, h_pool2_flat_4, h_pool2_flat_5, h_pool2_flat_6, h_pool2_flat_7, h_pool2_flat_8, h_pool2_flat_9, h_pool2_flat_10, h_pool2_flat_11, h_pool2_flat_12, h_pool2_flat_13, h_pool2_flat_14, h_pool2_flat_15, h_pool2_flat_16], 1)

    #h_pool2_flat = tf.reshape(h_pool2_flat, [-1, 8 * 8 * 16 * 16])

    h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## func2 layer ##
    W_fc2 = weight_variable([1024, 512], weight_name="fc2")
    b_fc2 = bias_variable([512], bias_name="fb2")

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    ## func3 layer ##
    W_fc3 = weight_variable([512, 1], weight_name="fc3")
    b_fc3 = bias_variable([1], bias_name="fb3")
    fc3 = tf.matmul(h_fc2_drop, W_fc3)

    y = tf.add(fc3, b_fc3, name="y")
    return y

def backward():
    datasets, label, test_data, test_label = reload_choose_duck_split()
    X = tf.placeholder(tf.float32, [None, 16, 40, 40], name="X")
    Y_ = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    y = forward(X, keep_prob)
    global_step = tf.Variable(0, trainable=False)
    loss_mse = tf.reduce_mean(tf.square(y - Y_))
    train_step = tf.train.AdamOptimizer(0.00001).minimize(loss_mse)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 训练模型。
        STEPS = 50000000001
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 1930   #duck6 3440; duck3 2021
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={X: datasets[start:end], Y_: label[start:end], keep_prob: 0.5})
            if i % 200 == 0:
                train_loss = sess.run(loss_mse, feed_dict={X: datasets, Y_: label, keep_prob: 1})
                total_loss = sess.run(loss_mse, feed_dict={X: test_data, Y_: test_label, keep_prob: 1})
                print("After %d training step(s), loss_mse on train data is %g, loss_mse on val data is %g" % (
                i, train_loss, total_loss))
            if i % 10000 == 0:
                saver.save(sess, './checkpoint/variable', global_step=i)

def main():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    backward()


if __name__ == '__main__':
    main()
