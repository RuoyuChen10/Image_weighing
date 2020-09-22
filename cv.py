import cv2
import numpy as np
vc = cv2.VideoCapture('./dark.MP4') #读入视频文件
c=1
if vc.isOpened(): #判断是否正常打开
    rval , frame = vc.read()
else:
    rval = False
timeF = 50 #视频帧计数间隔频率
t=1
while rval: #循环读取视频帧
    rval, frame = vc.read()
    if(c%timeF == 0): #每隔timeF帧进行存储操作
        frame=np.rot90(frame)
        frame=np.rot90(frame)
        frame=np.rot90(frame)
        cv2.imwrite('./tu//'+str(t) + '.jpg',frame) #存储为图像
        print('image/0'+str(t) + '.jpg')
        t=t+1
    c = c + 1
cv2.waitKey(1)
vc.release()