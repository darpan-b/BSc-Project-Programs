import cv2 as cv
import numpy as np

def display_image(curimg, imgname):
    resized_image = cv.resize(curimg, (600, 700), interpolation=cv.INTER_AREA)
    cv.imshow(imgname, resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_contours(img):
    contours, hierarchy  = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    print("Number of contours = ", len(contours))
    new_contours = []
    # TOT_AREA = len(img) * len(img[0])
    # for e in contours:
    #     x,y,w,h = cv.boundingRect(e)
    #     # print("x = ", x , " y = ", y, " w = ", w, " h = ", h, "e = ", e)
    #     comp_ratio = (w*h) / TOT_AREA
    #     print("comp_Ratio = ", comp_ratio)
    #     if comp_ratio >= 0.00005 and comp_ratio <= 0.08:
    #         new_contours.append([x,y,w,h])
    # contours = np.array(new_contours)
    res_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.drawContours(res_img, contours, -1, (0,255,0), 2)
    cv.imwrite("contours_drawn_img.png", res_img)
    return contours, res_img

cropped_img = cv.imread('cropped.png',0)
ret, bin_image = cv.threshold(cropped_img, 155, 255, cv.THRESH_BINARY)
contours_arr, contours_img = find_contours(bin_image)

rect_img = np.array(contours_img)

for e in contours_arr:
    cnt = e

    # compute straight bounding rectangle
    x,y,w,h = cv.boundingRect(cnt)
    # rect_img = cv.drawContours(rect_img,[cnt],0,(255,255,0),2)
    rect_img = cv.rectangle(rect_img,(x,y),(x+w,y+h),(255,0,0),2)

cv.imwrite("rectangles_drawn_over_contours.png", rect_img)

circ_img = np.array(contours_img)

for e in contours_arr:
    cnt = e
    
    (x_axis,y_axis),radius = cv.minEnclosingCircle(cnt) 
    
    center = (int(x_axis),int(y_axis)) 
    radius = int(radius) 
    
    cv.circle(circ_img,center,radius,(0,0,255),2) 

cv.imwrite("circles_drawn_over_contours.png", circ_img)
