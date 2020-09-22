import cv2
import os
for i in range(1,51):
    dirs = os.listdir("./duck/" + str(i))
    for pic_name in dirs:
        pic_dir = "duck/" + str(i) + "/" + pic_name
        save_dir = "duck2/" + str(i) + "/" + pic_name
        img = cv2.imread(pic_dir)
        img = cv2.resize(img, (50, 214))
        cv2.imwrite(save_dir, img)
        pic_dir = None
        save_dir = None
    print('finish ' + str(i) + '/50')