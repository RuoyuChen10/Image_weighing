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
    initial = tf.constant(2.0, shape=shape)
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
    with tf.name_scope('parameters'):
        ## convl layer ##
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([3, 3, 16, 4], weight_name="conv1")  # kernel 3*3, channel is 3
            tf.summary.histogram('conv1', W_conv1)
        with tf.name_scope('cb1'):
            b_conv1 = bias_variable([4], bias_name="cb1")
            tf.summary.histogram('cb1', b_conv1)
        with tf.name_scope('h_conv1'):
            h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)  # output size 38*38*8
            tf.summary.histogram('h_conv_1', h_conv1)
        with tf.name_scope('h_pool1'):
            h_pool1 = avg_pool_2x2(h_conv1)  # output size 19*19*8
            tf.summary.histogram('h_pool1', h_pool1)

        ## conv2 layer ##
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([3, 3, 4, 8], weight_name="conv2")  # kernel 3*3, in size 3, out size 5
            tf.summary.histogram('conv2', W_conv2)
        with tf.name_scope('cb2'):
            b_conv2 = bias_variable([8], bias_name="cb2")
            tf.summary.histogram('cb2', b_conv2)
        with tf.name_scope('h_conv2'):
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 17*17*8
            tf.summary.histogram('h_conv2', h_conv2)

        ## funcl layer ##
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([17*17*8, 1024], weight_name="fc1")
            tf.summary.histogram('fc1', W_fc1)
        with tf.name_scope('fb1'):
            b_fc1 = bias_variable([1024], bias_name="fb1")
            tf.summary.histogram('fb1', b_fc1)

        # [n_samples,7,7,64]->>[n_samples, 7*7*64]
        with tf.name_scope('h_pool2_flat'):
            h_pool2_flat = tf.reshape(h_conv2, [-1, 17*17*8])
            tf.summary.histogram('h_pool2_flat', h_pool2_flat)
        with tf.name_scope('h_fc1'):
            h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            tf.summary.histogram('h_fc1', h_fc1)
        with tf.name_scope('h_fc1_drop'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            tf.summary.histogram('h_fc1_drop', h_fc1_drop)

        ## func2 layer ##
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([1024, 512], weight_name="fc2")
            tf.summary.histogram('fc2', W_fc2)
        with tf.name_scope('fb2'):
            b_fc2 = bias_variable([512], bias_name="fb2")
            tf.summary.histogram('fb2', b_fc2)

        with tf.name_scope('h_fc2'):
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            tf.summary.histogram('h_fc2', h_fc2)
        with tf.name_scope('h_fc2_drop'):
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
            tf.summary.histogram('h_fc2_drop', h_fc2_drop)

        ## func3 layer ##
        with tf.name_scope('fc3'):
            W_fc3 = weight_variable([512, 1], weight_name="fc3")
            tf.summary.histogram('fc3', W_fc3)
        with tf.name_scope('fb3'):
            b_fc3 = bias_variable([1], bias_name="fb3")
            tf.summary.histogram('fb3', b_fc3)
        with tf.name_scope('mfc3'):
            fc3 = tf.matmul(h_fc2_drop, W_fc3)
            tf.summary.histogram('mfc3', fc3)
        with tf.name_scope('y'):
            y = tf.add(fc3, b_fc3, name="y")
            tf.summary.histogram('y', y)
    return y

def backward():
    #datasets, label, test_data, test_label = Distribution_superposition()
    datasets, label, test_data, test_label = reload_choose_duck_fold()
    with tf.name_scope('data'):
        X = tf.placeholder(tf.float32, [None, 40, 40, 16], name="X")
        Y_ = tf.placeholder(tf.float32, [None, 1])
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        tf.summary.scalar('keep_prob', keep_prob)
    y = forward(X, keep_prob)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y - Y_))
        tf.summary.scalar('loss', loss)

    LEARNING_RATE_BASE = 0.00001  # 最初学习率
    LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
    LEARNING_RATE_STEP = 10000  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/BATCH_SIZE
    gloabl_steps = tf.Variable(0, trainable=False)  # 计数器，用来记录运行了几轮的BATCH_SIZE，初始为0，设置为不可训练
    with tf.name_scope('learning_rate'):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, gloabl_steps, LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_mse)
    saver = tf.train.Saver()
    merged_summary_op = tf.summary.merge_all()
    with tf.name_scope('init_op'):
        init_op = tf.global_variables_initializer()
    min_loss = 1
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('./log', sess.graph)
        sess.run(init_op)
        # 训练模型。
        STEPS = 50000000001
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 1930   #duck6 3440; duck3 2021
            end = start + BATCH_SIZE
            sess.run(train, feed_dict={X: datasets[start:end], Y_: label[start:end], keep_prob: 0.8})
            if i % 200 == 0:
                train_loss = sess.run(loss, feed_dict={X: datasets, Y_: label, keep_prob: 1})
                total_loss = sess.run(loss, feed_dict={X: test_data, Y_: test_label, keep_prob: 1})
                if total_loss<min_loss:
                    min_loss = total_loss
                print("After %d training step(s), loss_mse on train data is %g, loss_mse on val data is %g, min_loss is %g" % (
                i, train_loss, total_loss, min_loss))
            if i % 10000 == 0:
                saver.save(sess, './checkpoint/variable', global_step=i)
                summary_str = sess.run(merged_summary_op,feed_dict={X: datasets, Y_: label, keep_prob: 1})
                summary_writer.add_summary(summary_str, i)
            # if (i + 1) % 50000 == 0:
            #     datasets, label = dataset_disruption(datasets, label)


def main():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    backward()


if __name__ == '__main__':
    main()
