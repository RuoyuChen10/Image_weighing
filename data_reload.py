import cv2
import numpy as np
import xlrd
import random
import os

# tolist
def get_data():
    # 此函数读取datasets文件夹下的图片，并和excel表格的标签一一对应
    # 返回数据与标签，单个数据为1维向量形式，非图片的矩阵形式
    data = []
    label = []
    # 先得到标签集
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
    # 得到图片数据
    for i in range(1,21):
        os_dir = './datasets/' + str(i) +'/'
        for j in range(1,5):
            if(i<10):
                pic_dir = os_dir + '0' + str(i) +'0' + str(j) + '.jpg'
            else:
                pic_dir = os_dir + str(i) +'0' + str(j) + '.jpg'
            img = cv2.imread(pic_dir, 0)
            img = cv2.resize(img, (104,432))
            li = img.tolist()
            one_data = []
            for k in range(0, len(li)):
                one_data = one_data + li[k]
            data.append(one_data)
            if(i<=10):
                label_c = 0
                label_r = i-1+2
            else:
                label_c = 1
                label_r = i-1+2-10
            label.append([float(xls_list[label_r][label_c])])
    return data, label

def random_choose():
    # 此函数配合get_data函数，随机读取并分配至训练集与测试集
    data, label = get_data()
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    num = len(data)
    train_num = int(0.8 * num)
    su = []
    for i in range(0, num):
        su.append(i)
    num_train = sorted(random.sample(su, train_num))
    for i in range(0, num):
        if(int(i) in num_train):
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
    return train_data, train_label, test_data, test_label

def Sequential_disruption(train_data, train_label, test_data, test_label):
    # 此函数对训练集与数据集进行随机的打乱，防止其整齐
    train_num = len(train_label)
    test_num = len(test_label)
    train_seq_distruption = [i for i in range(0, train_num)]
    test_seq_distruption = [i for i in range(0, test_num)]
    random.shuffle(train_seq_distruption)
    random.shuffle(test_seq_distruption)
    distrupted_train_data = []
    distrupted_train_label = []
    distrupted_test_data = []
    distrupted_test_label = []
    for i in train_seq_distruption:
        distrupted_train_data.append(train_data[i])
        distrupted_train_label.append(train_label[i])
    for i in test_seq_distruption:
        distrupted_test_data.append(test_data[i])
        distrupted_test_label.append(test_label[i])
    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label

def reload_all_data():
    # 此函数依次读取全部的鸭子图片，并直接分为训练集与测试集，数据仍为一维的。
    data = []
    label = []
    # 先得到标签集
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
    # 得到图片数据
    for i in range(1,51):
        dirs = os.listdir("./duck/"+str(i))
        for pic_name in dirs:
            pic_dir = "duck/" + str(i) + "/" +pic_name
            img = cv2.imread(pic_dir, 0)
            img = cv2.resize(img, (104, 432))
            li = img.tolist()
            one_data = []
            for k in range(0, len(li)):
                one_data = one_data + li[k]
            data.append(one_data)
            label.append([float(xls_list[int((i-1)%10+2)][int((i-1)/10)] )])
            pic_dir = None
        print('finish load ' + str(i) + '/50 data')
    # 现在开始对数据进行随机分配，制作训练集与数据集。
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    num = len(data)
    train_num = int(0.8 * num)
    su = []
    for i in range(0, num):
        su.append(i)
    num_train = sorted(random.sample(su, train_num))
    for i in range(0, num):
        if (int(i) in num_train):
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
    print("data reload OK!")
    distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label = Sequential_disruption(train_data, train_label, test_data, test_label)
    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label

def reload_all_data_2D():
    # 此函数依次读取全部的鸭子图片，并直接分为训练集与测试集，数据仍为一维的。
    data = []
    label = []
    # 先得到标签集
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
    # 得到图片数据
    for i in range(1, 41):
        dirs = os.listdir("./duck2/" + str(i))
        for pic_name in dirs:
            pic_dir = "duck2/" + str(i) + "/" + pic_name
            img = cv2.imread(pic_dir)
            data.append(img)
            label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
            pic_dir = None
        print('finish load ' + str(i) + '/50 data')
    # 现在开始对数据进行随机分配，制作训练集与数据集。
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    num = len(data)
    train_num = int(0.8 * num)
    su = []
    for i in range(0, num):
        su.append(i)
    num_train = sorted(random.sample(su, train_num))
    for i in range(0, num):
        if (int(i) in num_train):
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
    print("data reload OK!")
    distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label = Sequential_disruption(train_data, train_label, test_data, test_label)

    distrupted_train_data = np.array(distrupted_train_data)
    distrupted_train_data = distrupted_train_data.astype(np.float32)

    distrupted_train_label = np.array(distrupted_train_label)
    distrupted_train_label = distrupted_train_label.astype(np.float32)

    distrupted_test_data = np.array(distrupted_test_data)
    distrupted_test_data = distrupted_test_data.astype(np.float32)

    distrupted_test_label = np.array(distrupted_test_label)
    distrupted_test_label = distrupted_test_label.astype(np.float32)

    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label

def reload_all_data_R():
    # 此函数依次读取全部的鸭子图片，并直接分为训练集与测试集，数据仍为一维的。
    data = []
    label = []
    # 先得到标签集
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
    # 得到图片数据
    for i in range(1, 51):
        dirs = os.listdir("./duck3/" + str(i))
        for pic_name in dirs:
            pic_dir = "duck3/" + str(i) + "/" + pic_name
            img = cv2.imread(pic_dir)
            b, g, r = cv2.split(img)
            data.append(r)
            label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
            pic_dir = None
        print('finish load ' + str(i) + '/50 data')
    # 现在开始对数据进行随机分配，制作训练集与数据集。
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    num = len(data)
    train_num = int(0.8 * num)
    su = []
    for i in range(0, num):
        su.append(i)
    num_train = sorted(random.sample(su, train_num))
    for i in range(0, num):
        if (int(i) in num_train):
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
    print("data reload OK!")
    distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label = Sequential_disruption(train_data, train_label, test_data, test_label)

    distrupted_train_data = np.array(distrupted_train_data)
    distrupted_train_data = distrupted_train_data.astype(np.float32)

    distrupted_train_label = np.array(distrupted_train_label)
    distrupted_train_label = distrupted_train_label.astype(np.float32)

    distrupted_test_data = np.array(distrupted_test_data)
    distrupted_test_data = distrupted_test_data.astype(np.float32)

    distrupted_test_label = np.array(distrupted_test_label)
    distrupted_test_label = distrupted_test_label.astype(np.float32)

    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label

def reload_choose_duck_R(contral = None):
    # 此函数依次读取全部的鸭子图片，数据仍为一维的,对50只鸭子中抽5只不作为训练。
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    if contral != None:
        #choose_duck_num = [9, 20, 23, 35, 45]
        choose_duck_num = [6, 8, 14, 19, 37]
    else:
        choose_duck_num = random.sample(range(11, 41), 5)    # 随机抽取的鸭子代号
        choose_duck_num = sorted(choose_duck_num)
    print(choose_duck_num)
    # 先得到标签集
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
    # 得到图片数据
    for i in range(1, 51):
        dirs = os.listdir("./duck2/" + str(i))
        if i in choose_duck_num:
            for pic_name in dirs:
                pic_dir = "duck2/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                test_data.append(r)
                test_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
            print('finish load ' + str(i) + '/50 data')
        else:
            for pic_name in dirs:
                pic_dir = "duck2/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                train_data.append(r)
                train_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
            print('finish load ' + str(i) + '/50 data')
    # 现在开始对数据进行随机分配，制作训练集与数据集。

    distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label = Sequential_disruption(train_data, train_label, test_data, test_label)

    distrupted_train_data = np.array(distrupted_train_data)
    distrupted_train_data = distrupted_train_data.astype(np.float32)

    distrupted_train_label = np.array(distrupted_train_label)
    distrupted_train_label = distrupted_train_label.astype(np.float32)

    distrupted_test_data = np.array(distrupted_test_data)
    distrupted_test_data = distrupted_test_data.astype(np.float32)

    distrupted_test_label = np.array(distrupted_test_label)
    distrupted_test_label = distrupted_test_label.astype(np.float32)

    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label

def reload_choose_duck_R_no_background(contral = None):
    # 此函数依次读取全部的鸭子图片，数据仍为一维的,对50只鸭子中抽5只不作为训练。
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    if contral != None:
        choose_duck_num = [16, 20, 23, 35, 37]
        choose_duck_num = [20, 23, 37, 47, 48]
        #choose_duck_num = [9, 15, 21, 28, 50]
        #choose_duck_num = [6, 8, 14, 19, 37]
    else:
        #choose_duck_num = random.sample(range(1, 51), 5)    # 随机抽取的鸭子代号
        choose_duck_num = random.sample([34, 29, 43, 45, 46, 47, 48], 5)
        choose_duck_num = sorted(choose_duck_num)
    print(choose_duck_num)
    # 先得到标签集
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
    # 得到图片数据
    for i in range(1, 51):
        dirs = os.listdir("./duck2/" + str(i))
        if i in choose_duck_num:
            for pic_name in dirs:
                pic_dir = "duck2/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                thresh, img2 = cv2.threshold(r, 30, 0, cv2.THRESH_TOZERO)
                test_data.append(img2/255.)
                test_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
            print('finish load ' + str(i) + '/50 data')
        else:
            for pic_name in dirs:
                pic_dir = "duck2/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                thresh, img2 = cv2.threshold(r, 30, 0, cv2.THRESH_TOZERO)
                train_data.append(img2/255.)
                train_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
            print('finish load ' + str(i) + '/50 data')
    # 现在开始对数据进行随机分配，制作训练集与数据集。

    distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label = Sequential_disruption(train_data, train_label, test_data, test_label)

    distrupted_train_data = np.array(distrupted_train_data)
    distrupted_train_data = distrupted_train_data.astype(np.float32)

    distrupted_train_label = np.array(distrupted_train_label)
    distrupted_train_label = distrupted_train_label.astype(np.float32)

    distrupted_test_data = np.array(distrupted_test_data)
    distrupted_test_data = distrupted_test_data.astype(np.float32)

    distrupted_test_label = np.array(distrupted_test_label)
    distrupted_test_label = distrupted_test_label.astype(np.float32)

    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label

def reload_choose_duck_R_no_background_just_1(contral = None):
    # 此函数依次读取全部的鸭子图片，数据仍为一维的,对50只鸭子中抽5只不作为训练。
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    if contral != None:
        #choose_duck_num = [16, 20, 23, 35, 37]
        choose_duck_num = [22]
        #choose_duck_num = [9, 15, 21, 28, 50]
        #choose_duck_num = [6, 8, 14, 19, 37]
    else:
        choose_duck_num = random.sample(range(1, 51), 5)    # 随机抽取的鸭子代号
        choose_duck_num = sorted(choose_duck_num)
    print(choose_duck_num)
    # 先得到标签集
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
    # 得到图片数据
    for i in range(1, 51):
        dirs = os.listdir("./duck2/" + str(i))
        if i in choose_duck_num:
            for pic_name in dirs:
                pic_dir = "duck2/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                thresh, img2 = cv2.threshold(r, 30, 0, cv2.THRESH_TOZERO)
                test_data.append(img2)
                test_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
            print('finish load ' + str(i) + '/50 data')
        else:
            for pic_name in dirs:
                pic_dir = "duck2/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                thresh, img2 = cv2.threshold(r, 30, 0, cv2.THRESH_TOZERO)
                train_data.append(img2)
                train_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
            print('finish load ' + str(i) + '/50 data')
    # 现在开始对数据进行随机分配，制作训练集与数据集。

    distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label = Sequential_disruption(train_data, train_label, test_data, test_label)

    distrupted_train_data = np.array(distrupted_train_data)
    distrupted_train_data = distrupted_train_data.astype(np.float32)

    distrupted_train_label = np.array(distrupted_train_label)
    distrupted_train_label = distrupted_train_label.astype(np.float32)

    distrupted_test_data = np.array(distrupted_test_data)
    distrupted_test_data = distrupted_test_data.astype(np.float32)

    distrupted_test_label = np.array(distrupted_test_label)
    distrupted_test_label = distrupted_test_label.astype(np.float32)

    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label

def read_xls():
    # 先得到标签集
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
        return xls_list

def fold_picture(pic):
    data = []
    for i in range(0,8):
        for j in range(0,2):
            channel = []
            for k in range(0,40):
                channel.append(pic[i*40+k][j*40:j*40+40].tolist())
            data.append(channel)
            channel = None
    data = np.array(data)
    img2 = cv2.merge([i for i in data])
    return img2

def Distribution_superposition():
    data = []
    label = []
    xls_list = read_xls()
    # 得到图片数据
    for i in range(1, 51):
        dirs = os.listdir("./duck6/" + str(i))
        for pic_name in dirs:
            pic_dir = "duck6/" + str(i) + "/" + pic_name
            img = cv2.imread(pic_dir)
            b, g, r = cv2.split(img)
            img2 = fold_picture(r)
            data.append(img2)
            label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
            pic_dir = None
            img2 = None
        print('finish load ' + str(i) + '/50 data')
    # 现在开始对数据进行随机分配，制作训练集与数据集。
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    num = len(data)
    train_num = int(0.8 * num)
    su = []
    for i in range(0, num):
        su.append(i)
    num_train = sorted(random.sample(su, train_num))
    for i in range(0, num):
        if (int(i) in num_train):
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
    print("data reload OK!")

    distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label = Sequential_disruption(train_data, train_label, test_data, test_label)

    distrupted_train_data = np.array(distrupted_train_data)
    distrupted_train_data = distrupted_train_data.astype(np.float32)

    distrupted_train_label = np.array(distrupted_train_label)
    distrupted_train_label = distrupted_train_label.astype(np.float32)

    distrupted_test_data = np.array(distrupted_test_data)
    distrupted_test_data = distrupted_test_data.astype(np.float32)

    distrupted_test_label = np.array(distrupted_test_label)
    distrupted_test_label = distrupted_test_label.astype(np.float32)

    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label

def reload_choose_duck_fold(contral = None):
    # 此函数依次读取全部的鸭子图片，数据仍为一维的,对50只鸭子中抽5只不作为训练。
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    if contral != None:
        choose_duck_num = [6, 10, 47]
    else:
        choose_duck_num = random.sample(range(1, 51), 5)    # 随机抽取的鸭子代号
        choose_duck_num = sorted(choose_duck_num)
    print(choose_duck_num)
    # 先得到标签集
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
    # 得到图片数据
    for i in range(1, 51):
        dirs = os.listdir("./duck3/" + str(i))
        if i in choose_duck_num:
            for pic_name in dirs:
                pic_dir = "duck3/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                thresh, img2 = cv2.threshold(r, 60, 0, cv2.THRESH_TOZERO)
                fold_img = fold_picture(img2)
                test_data.append(fold_img)
                test_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
                fold_img = None
            print('finish load ' + str(i) + '/50 data')
        else:
            for pic_name in dirs:
                pic_dir = "duck3/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                thresh, img2 = cv2.threshold(r, 60, 0, cv2.THRESH_TOZERO)
                fold_img = fold_picture(img2)
                train_data.append(fold_img)
                train_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
            print('finish load ' + str(i) + '/50 data')
    # 现在开始对数据进行随机分配，制作训练集与数据集。

    distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label = Sequential_disruption(train_data, train_label, test_data, test_label)

    distrupted_train_data = np.array(distrupted_train_data)
    distrupted_train_data = distrupted_train_data.astype(np.float32)

    distrupted_train_label = np.array(distrupted_train_label)
    distrupted_train_label = distrupted_train_label.astype(np.float32)

    distrupted_test_data = np.array(distrupted_test_data)
    distrupted_test_data = distrupted_test_data.astype(np.float32)

    distrupted_test_label = np.array(distrupted_test_label)
    distrupted_test_label = distrupted_test_label.astype(np.float32)

    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label

########################################################################################################################
########################################################################################################################
########################################################################################################################

def split_picture(pic):
    data = []
    for i in range(0,8):
        for j in range(0,2):
            channel = []
            for k in range(0,40):
                channel.append(pic[i*40+k][j*40:j*40+40].tolist())
            data.append(channel)
    data = np.array(data)
    x1 = data[0]
    x2 = data[1]
    x3 = data[2]
    x4 = data[3]
    x5 = data[4]
    x6 = data[5]
    x7 = data[6]
    x8 = data[7]
    x9 = data[8]
    x10 = data[9]
    x11 = data[10]
    x12 = data[11]
    x13 = data[12]
    x14 = data[13]
    x15 = data[14]
    x16 = data[15]
    #return x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16
    return data

def reload_choose_duck_split(contral = None):
    # 此函数依次读取全部的鸭子图片，数据仍为一维的,对50只鸭子中抽5只不作为训练。
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    if contral != None:
        choose_duck_num = [6, 10, 47]
    else:
        choose_duck_num = random.sample(range(1, 51), 5)    # 随机抽取的鸭子代号
        choose_duck_num = sorted(choose_duck_num)
    print(choose_duck_num)
    # 先得到标签集
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
    # 得到图片数据
    for i in range(1, 51):
        dirs = os.listdir("./duck3/" + str(i))
        if i in choose_duck_num:
            for pic_name in dirs:
                pic_dir = "duck3/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                thresh, img2 = cv2.threshold(r, 60, 0, cv2.THRESH_TOZERO)
                fold_img = split_picture(img2)
                test_data.append(fold_img)
                test_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
                fold_img = None
            print('finish load ' + str(i) + '/50 data')
        else:
            for pic_name in dirs:
                pic_dir = "duck3/" + str(i) + "/" + pic_name
                img = cv2.imread(pic_dir)
                b, g, r = cv2.split(img)
                thresh, img2 = cv2.threshold(r, 60, 0, cv2.THRESH_TOZERO)
                fold_img = split_picture(img2)
                train_data.append(fold_img)
                train_label.append([float(xls_list[int((i - 1) % 10 + 2)][int((i - 1) / 10)])])
                pic_dir = None
            print('finish load ' + str(i) + '/50 data')
    # 现在开始对数据进行随机分配，制作训练集与数据集。

    distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label = Sequential_disruption(train_data, train_label, test_data, test_label)

    distrupted_train_data = np.array(distrupted_train_data)
    distrupted_train_data = distrupted_train_data.astype(np.float32)

    distrupted_train_label = np.array(distrupted_train_label)
    distrupted_train_label = distrupted_train_label.astype(np.float32)

    distrupted_test_data = np.array(distrupted_test_data)
    distrupted_test_data = distrupted_test_data.astype(np.float32)

    distrupted_test_label = np.array(distrupted_test_label)
    distrupted_test_label = distrupted_test_label.astype(np.float32)

    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label