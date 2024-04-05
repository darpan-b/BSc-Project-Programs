# aim is to check how many straight lines there are
import cv2 as cv
import numpy as np

def display_image(curimg, imgname):
    resized_image = cv.resize(curimg, (600, 700), interpolation=cv.INTER_AREA)
    cv.imshow(imgname, resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


cropped_img = cv.imread('cropped.png',0)
ret, bin_image = cv.threshold(cropped_img, 155, 255, cv.THRESH_BINARY)




working_image = cv.bitwise_not(bin_image)
# display_image(cropped_img, 'cropped_img')
# display_image(bin_image, 'bin image')
display_image(working_image, 'working_image')

''' PROBABLY THE MASK SIZE AND IMAGE SIZE NEEDS TO BE SAME '''

# mask = np.zeros((5,15), np.uint8)
mask = np.zeros(working_image.shape[:2], dtype=np.uint8)
mask[:, :] = 0
# mask[1:3, 0:15] = 255 ### ei figure ta loop er moddhe modify korte jete jobe, aager line tao include korte hobe

# mask = mask.astype(np.uint8)
# cv.imwrite("white_mask.jpg", mask) 
cv.imwrite("black_mask.png", mask)
# masked_img = cv.bitwise_and(cropped_img, cropped_img, mask=mask)

# display_image(mask, 'mask')
# display_image(masked_img, 'masked image')

# working_img = np.array(bin_image)

# print("mask shape", mask.shape, "working image shape", working_img.shape)

# display_image(working_img, 'working img')

working_image2 = np.array(working_image)

'''
puro 1 ghonta time laglo run korte sala
for i in range(80,len(working_image)-5,5):
    for j in range(0,len(working_image)-15,15):
        mask[i:i+5, j:j+15] = 255
        masked_img = cv.bitwise_and(working_image, working_image, mask=mask)
        white_pix_count = np.sum(masked_img==255)
        if white_pix_count > 50:
            working_image2[i:i+5, j:j+15] = 255
        else:
            working_image2[i:i+5, j:j+15] = 0
            # print("i ", i, "j ", j, "white pix count = ", white_pix_count)
        mask[i:i+5, j:j+15] = 0

cv.imwrite("straight_lines_img.png", working_image2)
'''


'''
TRYING A MANUAL APPROACH

1 x row length er matrix diye try kori
after all that is found we'll count number of connected components
and compare the number of connected components for finding a metric
'''

# LINE COUNT = 50






