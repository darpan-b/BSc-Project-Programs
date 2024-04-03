# aim is to check how many straight lines there are
import cv2 as cv
import numpy as np

def display_image(curimg, imgname):
    resized_image = cv.resize(curimg, (600, 700), interpolation=cv.INTER_AREA)
    cv.imshow(imgname, resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


cropped_img = cv.imread('cropped.png',0)

mask = np.zeros((5,5), np.uint8)
mask[:, :] = 255
mask[1:3, 0:5] = 0
cv.imwrite("white_mask.jpg",mask) 
# masked_img = cv.bitwise_and(cropped_img, cropped_img, mask=mask)

display_image(mask, 'mask')
# display_image(masked_img, 'masked image')



