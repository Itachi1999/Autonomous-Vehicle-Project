import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_scale, (5, 5), 0)
    gradient = cv2.Canny(blur, 50, 150)
    return gradient


def make_coordinates(image, parameters):
    slope, intercept = parameters
    slope = float(slope)
    intercept = float(intercept)
    y1 = image.shape[0]
    y2 = int(y1*(3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    try:       
        left_avg = np.average(left_fit, axis = 0)
        right_avg = np.average(right_fit, axis = 0)
        right_co = make_coordinates(image, right_avg)
        left_co = make_coordinates(image, left_avg)
        return np.array([left_co, right_co])
    except Exception as e: print(e, '\n') #print error to console return None

def display_lines(image, lines):
    dummy = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(dummy, (x1, y1), (x2, y2), (0, 255, 0), 10)
    
    return dummy

def region_of_interest(gradient):
    height = 700
    triangle = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])
    mask = np.zeros_like(gradient)
    cv2.fillPoly(mask, triangle, 255)
    masked_img = cv2.bitwise_and(gradient, mask)
    return masked_img

'''image = cv2.imread('test_image.jpg')
lane_img = np.copy(image)
gradient = canny(lane_img)
cropped_image = region_of_interest(gradient)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
avg_lines = average_slope_intercept(lane_img, lines)
lined_image = display_lines(lane_img, avg_lines)
output = cv2.addWeighted(lane_img, 0.8, lined_image, 1, 1)
cv2.imshow('Result', output)
cv2.waitKey(0)'''


cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    gradient = canny(frame)
    cropped_image = region_of_interest(gradient)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    avg_lines = average_slope_intercept(frame, lines)
    lined_image = display_lines(frame, avg_lines)
    output = cv2.addWeighted(frame, 0.8, lined_image, 1, 1)
    cv2.imshow('Result', output)
    if  cv2.waitKey(1) == ord('c'):
        break
cap.release()
cv2.destroyAllWindows()
    
