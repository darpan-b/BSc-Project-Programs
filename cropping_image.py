''' CURRENT OBJECTIVE: Eliminating the red part from the datasheet '''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def display_image(curimg, imgname):
    resized_image = cv.resize(curimg, (600, 700), interpolation=cv.INTER_AREA)
    cv.imshow(imgname, resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


raw_img = cv.imread(r'C:\Users\DARPAN\Documents\College\6th Semester\BSc Project (DSE6)\Data\350_0018.png')

img_copy = np.array(raw_img)
img_copy2 = np.array(raw_img)

red_lower_bound = np.array([0, 0, 200])
red_upper_bound = np.array([255, 255, 255])

red_mask = cv.inRange(raw_img, red_lower_bound, red_upper_bound) # this is a binary image
red_mask2 = np.array(red_mask)

# display_image(red_mask, 'red_mask')

kernel = np.ones((2, 2), np.uint8)
kernel2 = np.ones((12, 12), np.uint8)
dilation = cv.dilate(red_mask2, kernel, iterations=4)
erosion = cv.erode(dilation, kernel2, iterations=15)

cv.imwrite('red_mask2.png', dilation)
cv.imwrite('noise_removed.png', erosion)


''' erosion theke coordiantes tola '''

contours, hierarchy = cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

lx, ly, lw, lh = 0, 0, 0, 0

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    # print('x, y, w, h', x, y, w, h)
    if x + y == 0: continue
    if lw * lh < w * h:
        lx, ly, lw, lh = x, y, w, h

cropped = red_mask[ly:ly+lh, lx:lx+lw]

cv.imwrite('cropped.png', cropped)

rect = cv.rectangle(img_copy2, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
