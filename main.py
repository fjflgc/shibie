import cv2 as cv
import numpy as np

#https://blog.csdn.net/m0_65833575/article/details/126423908
if __name__ == '__main__':
    # 实例化控制摄像头的类
    cap = cv.VideoCapture(1)
    #cap = cv.VideoCapture("image/1.mp4")
    # cap.isOpened() 摄像头初始化，返回true则成功
    while cap.isOpened():
        # 读取一帧图像，ret ：是否读取正确 ， frame ： 读取的图像
        ret, frame = cap.read()
        # 加载模型文件
        face = cv.CascadeClassifier('./data/haarcascade_eye.xml')
        # 执行模型，左上顶点（x，y）和w（宽），h（高）
        faces = face.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
        # 循环遍历识别到的每一张人脸数据
        for x, y, w, h in faces:
            # 在人脸上加马赛克
            frameBox = frame[y:y + h, x:x + w]
            frameBox = frameBox[::10, ::10]
            frameBox = np.repeat(frameBox, 10, axis=0)
            frameBox = np.repeat(frameBox, 10, axis=1)
            a, b = frame[y:y + h, x:x + w].shape[:2]
            frame[y:y + h, x:x + w] = frameBox[:a, :b]
            # 在人脸周围画框
            cv.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 255], 2)
        # 展示照片
        cv.imshow('frame', frame)
        # 1ms 读取一帧
        k = cv.waitKey(1)
        print(k)
        # 按“q”退出
        if k == ord('q'):
            break

    cv.destroyAllWindows()