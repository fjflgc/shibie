import cv2
import face_recognition
import numpy as np

# 加载示例视频
video_capture = cv2.VideoCapture(0)

while True:
    # 读取视频帧
    ret, frame = video_capture.read()
    if not ret:
        break

    # 使用Face Recognition库检测人脸及关键点
    face_landmarks_list = face_recognition.face_landmarks(frame)

    for face_landmarks in face_landmarks_list:
        # 获取嘴部区域的关键点
        mouth_points = face_landmarks['top_lip'] + face_landmarks['bottom_lip']
        mouth_points = np.array(mouth_points, dtype=np.int32)

        # 创建一个与视频帧大小相同的空白图像
        mask = np.zeros_like(frame)

        # 使用多边形填充嘴部区域
        cv2.fillPoly(mask, [mouth_points], (255, 255, 255))

        # 对嘴部区域进行高斯模糊处理，实现模糊效果
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # 将模糊的嘴部区域覆盖到原始视频帧上
        frame = cv2.bitwise_and(frame, mask)

    # 显示处理后的视频帧
    cv2.imshow('Video', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭窗口
video_capture.release()
cv2.destroyAllWindows()