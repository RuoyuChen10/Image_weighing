import tensorflow as tf
import numpy as np
import cv2
import train
import xlwt
import xlrd

ckeckpoint_dir = './checkpoint'

def pic_init(pic):
    image_dir = './test/' + str(pic) + '.jpg'
    image = cv2.imread(image_dir, 0)
    image = cv2.resize(image, (104,432))
    pic_list = image.tolist()
    one_data = []
    for k in range(0, len(pic_list)):
        one_data = one_data + pic_list[k]
    predict_data = []
    predict_data.append(one_data)
    return predict_data

def real_data_init():
    workbook = xlrd.open_workbook('weight.xls')
    Data_sheet = workbook.sheets()[0]
    rowNum = Data_sheet.nrows
    colNum = Data_sheet.ncols
    xls_list = []
    for i in range(rowNum):
        rowlist = []
        for j in range(colNum):
            rowlist.append(Data_sheet.cell_value(i, j))
        xls_list.append(rowlist)
    real_data = []
    for i in range(1,51):
        c = int((i-1)/10)
        r = (i-1)%10 + 2
        real_data.append(xls_list[r][c])
    return real_data

def main():
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('sheet1')
    ws.write(0, 0, '序号')
    ws.write(0, 1, '预测值')
    ws.write(0, 2, '真实值')
    ws.write(0, 3, '误差')
    real_data = real_data_init()
    for i in range(1,51):
        if(i!=13):
            test_pic = pic_init(i)
            sess = tf.Session()
            saver = tf.train.import_meta_graph('./checkpoint/variable-45600.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")
            feed_dict={X:test_pic}
            result = graph.get_tensor_by_name("y:0")
            a = sess.run(result, feed_dict)
            b = a[0]
            pre_weight = b[0]
            ws.write(i,1,float(pre_weight))
            loss = abs(float(pre_weight) - float(real_data[i-1]))
            ws.write(i, 3, loss)
            pre_weight = None
        ws.write(i, 0, int(i))
        ws.write(i, 2, real_data[i-1])
    wb.save('report.xls')
    print('Save the report.xls')


if __name__ == '__main__':
    main()