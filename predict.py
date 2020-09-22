import tensorflow as tf
import numpy as np
import cv2
import train
import os

ckeckpoint_dir = './checkpoint1'
# def forward(x):
#     # X = tf.placeholder(tf.float32, shape=(None, 44928))
#     # Y = tf.placeholder(tf.float32, shape=(None, 1))
#     w1 = get_weight([44928,1000], REGULARIZER, weight_name = 'w1')
#     w2 = get_weight([1000, 100], REGULARIZER, weight_name='w2')
#     w3 = get_weight([100,10], REGULARIZER, weight_name = 'w3')
#     w4 = get_weight([10,1], REGULARIZER, weight_name = 'w4')
#     b1 = get_bias([1000], 'b1')
#     b2 = get_bias([100], 'b2')
#     b3 = get_bias([10], 'b3')
#     b4 = get_bias([1], 'b4')
#     y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
#     y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
#     y3 = tf.nn.tanh(tf.matmul(y2, w3) + b3)
#     y = tf.matmul(y3, w4) + b4
#     return y

# def load_model_init():
#     with tf.Graph.as_default():
#         saver = tf.train.Saver()
#     with tf.Session(graph = tf.Graph) as sess:
#         sess.run(tf.global_variables_initializer())
#         saver.restore(sess, "./checkpoint/variable18000.ckpt")

def predict(image_dir):
    image = cv2.imread(image_dir)
    image = cv2.resize(image, (80,324))
    b, g, r = cv2.split(image)
    predict_data = []
    predict_data.append(r)
    predict_data = np.array(predict_data)
    predict_data = predict_data.astype(np.float32)
    return predict_data

def main():
    # load_model_init()
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./checkpoint1/variable-4547000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint1'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    result = graph.get_tensor_by_name("y:0")
    while True:
        image_dir = input("请输入您需要预测的鸭子的代号(输入数字‘0’结束程序):")
        if image_dir == '0':
            break
        try:
            predict_data = predict(image_dir)
            feed_dict={X:predict_data, keep_prob:1}
            print("这只鸭子预计重："+ str(sess.run(result, feed_dict)[0][0])+ 'kg')
        except:
            print('Open Error! Try again!')
            continue

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()