import numpy as np
import cv2

polygon_vertices = np.array([[(350,355),(590,355),(950,540),(40,540)]])
window_perspective = np.array([[(0,0),(480,0),(0,600),(480,600)]])

def region_of_interest(image):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon_vertices,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

capture = cv2.VideoCapture(r"C:\Users\chand\Downloads\whiteline.mp4")
while True:
    IsTrue, frame = capture.read()
    # frame = cv2.flip(image, 1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    L_roi = region_of_interest(gray)
    blur = cv2.GaussianBlur(L_roi,(3,3),0)
    ret, thresh1 = cv2.threshold(L_roi, 180, 255, cv2.THRESH_BINARY)

    my_frame = region_of_interest(thresh1)
    canny = cv2.Canny(my_frame,10,150)

    points_1 = np.float32([[(350,355),(590,355),(40,frame.shape[0]),(frame.shape[1],frame.shape[0])]])
    points_2 = np.float32([[(0,0),(480,0),(0,600),(480,600)]])
    matrix = cv2.getPerspectiveTransform(points_1,points_2)
    result = cv2.warpPerspective(canny,matrix,(480,600),flags = cv2.INTER_LANCZOS4)
    histogram = np.sum(result, axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftlanepixel_initial = np.sum(histogram[:midpoint])
    rightlanepixel_initial = np.sum(histogram[midpoint:])

    if leftlanepixel_initial > rightlanepixel_initial:
        left = (0, 255, 0)
        right = (0,0,255)
    else:
        left = (0, 0, 255)
        right = (0, 255, 0)

    left_lines = cv2.HoughLinesP(canny[:,0:480], 2, np.pi / 180, 100, np.array([]), minLineLength=70, maxLineGap=150)
    right_lines = cv2.HoughLinesP(canny[:,480:], 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)

    line_image = np.zeros_like(frame)
    if left_lines is not None:
        for line in left_lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1,y1),(x2,y2),left,10)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    if right_lines is not None:
        for line in right_lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1+480,y1),(x2+480,y2),right,10)
    combo_image1 = cv2.addWeighted(frame, 0.8, combo_image, 1, 1)
    cv2.imshow('frames', combo_image1)
    if cv2.waitKey(25) & 0xFF == ord('d'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()