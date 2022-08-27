import cv2
import numpy as np
import matplotlib.pyplot as plt

frame = cv2.imread('000295.jpg')
height, weigth = frame.shape[0], frame.shape[1]
# print(height,weigth)
last_mes = current_mes = np.array((0, height // 2), np.float32)  # 保存当前中心点，可替换为船舶检测出来的中心点坐标格式为[[x][y]]
last_pre = current_pre = np.array((0, height // 2), np.float32)  # 保存预测[[x][y][x误差][y误差]]


def mousemove(event, x, y, s, p):
    # x和y需要自己抛出来,中心点左边的x，y
    global frame, current_mes, last_mes, current_pre, last_pre

    last_pre = current_pre
    last_mes = current_mes

    current_mes = np.array([[np.float32(x)], [np.float32(y)]])

    kalman.correct(current_mes)
    current_pre = kalman.predict()

    lmx, lmy = last_mes[0], last_mes[1]
    lpx, lpy = last_pre[0], last_pre[1]
    cmx, cmy = current_mes[0], current_mes[1]
    cpx, cpy = current_pre[0], current_pre[1]
    cv2.line(frame, (int(lmx), int(lmy)), (int(cmx), int(cmy)), (0, 200, 0))  # 实际轨迹
    cv2.line(frame, (int(lpx), int(lpy)), (int(cpx), int(cpy)), (0, 0, 200))  # 预测轨迹


cv2.namedWindow("Kalman")
cv2.setMouseCallback("Kalman", mousemove)
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1

while (True):
    cv2.imshow('Kalman', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
