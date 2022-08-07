import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 80, 110)
    return canny

def roi(image):
    height = image.shape[0]
    polygons = np.array([
    [(70, 330 ), (253, 73), (608, 310)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image
image = cv2.imread('image.png')
lane_image = np.copy(image)

#log transformation
# c = 255 / np.log(1 + np.max(lane_image))
# log_image = c * (np.log(lane_image + 1))
# log_image = np.array(log_image, dtype = np.uint8)
#------------#

canny = canny(lane_image)
cropped_image = roi(canny)

lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=5)
line_image = display_lines(lane_image, lines)
blend = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv2.imshow("result", blend)
cv2.waitKey(0)

#plt.imshow(image)
#plt.show()

# plt.imshow(log_image)
# plt.show()