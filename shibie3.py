# https://blog.51cto.com/u_15872074/5841572
import cv2
import dlib
from imutils import face_utils
import numpy as np

predictor_path = "data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

border = 5
MOUSE_START = 49 - 1
MOUSE_END = 68 - 1
cap = cv2.VideoCapture(0)  # 如何参数为0，读取摄像头信息，如果为文件，读取视频

while (1):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        points = face_utils.shape_to_np(shape)
        mouse_points = points[MOUSE_START:MOUSE_END]
        mouseHull = cv2.convexHull(mouse_points)
        xr, yr, wr, hr = cv2.boundingRect(mouseHull)
        # 在人脸上加马赛克
        frameBox = img[yr:yr + hr, xr:xr + wr]
        frameBox = frameBox[::5, ::5]
        frameBox = np.repeat(frameBox, 5, axis=0)
        frameBox = np.repeat(frameBox, 5, axis=1)
        a, b = img[yr:yr + hr, xr:xr + wr].shape[:2]
        img[yr:yr + hr, xr:xr + wr] = frameBox[:a, :b]

        #cv2.rectangle(img, (xr - border, yr - border), (xr + wr + border, yr + hr + border), (0, 255, 9), 2)
        # cv2.drawContours(img, [mouseHull], -1, (0, 255, 0), 1)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC退出
        break
    cv2.imshow("Frame", img)

cap.release()
cv2.destroyAllWindows()