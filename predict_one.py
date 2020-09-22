import tensorflow as tf
import numpy as np
import cv2
import train
import xlwt
import xlrd
import os

ckeckpoint_dir = './checkpoint'
test_pic_dir = "./original_datasets/1/"

def pic_init(pic):
    image_dir = test_pic_dir + pic
    image = cv2.imread(image_dir)
    image = cv2.resize(image, (80, 324))
    b, g, r = cv2.split(image)
    predict_data = []
    predict_data.append(r)
    predict_data = np.array(predict_data)
    predict_data = predict_data.astype(np.float32)
    return predict_data

def main():
    dirs = os.listdir(test_pic_dir)
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('1')
    ws.write(0, 0, '序号')
    ws.write(0, 1, '预测值')
    ws.write(0, 2, '真实值')
    ws.write(0, 3, '误差')
    num = 1
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./checkpoint/variable-720000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    result = graph.get_tensor_by_name("y:0")
    for i in dirs:
        test_pic = pic_init(i)
        feed_dict = {X: test_pic, keep_prob: 1}
        a = sess.run(result, feed_dict)
        b = a[0]
        pre_weight = b[0]
        ws.write(num,0,i.split('.')[0])
        ws.write(num,1,float(pre_weight))
        pre_weight = None
        num = num + 1
    wb.save('单种报告.xls')
    print('Save the report.xls')


if __name__ == '__main__':
    main()