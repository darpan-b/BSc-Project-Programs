import cv2 as cv
import numpy as np
import math

def display_image(curimg, imgname):
    resized_image = cv.resize(curimg, (600, 700), interpolation=cv.INTER_AREA)
    cv.imshow(imgname, resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_contours(img):
    contours, hierarchy  = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    print("Number of contours = ", len(contours))
    res_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.drawContours(res_img, contours, -1, (0,255,0), 2)
    cv.imwrite("contours_drawn_img.png", res_img)
    return contours, res_img

cropped_img = cv.imread('cropped.png',0)
ret, bin_image = cv.threshold(cropped_img, 155, 255, cv.THRESH_BINARY)
contours_arr, contours_img = find_contours(bin_image)

rect_img = np.array(contours_img)

widths = np.array([])
heights = np.array([])
rratios = np.array([])

for e in contours_arr:
    cnt = e

    # compute straight bounding rectangle
    x,y,w,h = cv.boundingRect(cnt)
    # rect_img = cv.drawContours(rect_img,[cnt],0,(255,255,0),2)

    # print("w*h = ", w*h, "contour area = ", cv.contourArea(e))
    if w*h >= 500:
        rect_img = cv.rectangle(rect_img,(x,y),(x+w,y+h),(255,0,0),2)
        widths = np.append(widths, w)
        heights = np.append(heights, h)
        rratios = np.append(rratios, cv.contourArea(e)/(w*h))

cv.imwrite("rectangles_drawn_over_contours.png", rect_img)

circ_img = np.array(contours_img)

radiuses = np.array([])
cratios = np.array([])

for e in contours_arr:
    cnt = e
    
    (x_axis,y_axis),radius = cv.minEnclosingCircle(cnt) 
    
    center = (int(x_axis),int(y_axis)) 
    radius = int(radius) 
    if radius >= 20:
        cv.circle(circ_img,center,radius,(0,0,255),2) 
        radiuses = np.append(radiuses, radius)
        cratios = np.append(rratios, cv.contourArea(e)/(math.pi*radius*radius))

cv.imwrite("circles_drawn_over_contours.png", circ_img)

print("median height = ", np.median(heights))
print("median width = ", np.median(widths))
print("median radius = ", np.median(radiuses))

print("median rectangularity ratio = ", np.median(rratios))
print("median circularity ratio = ", np.median(cratios))

