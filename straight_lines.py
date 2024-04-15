import cv2 as cv
import numpy as np

def highlight_feature(mask, img):
    ''' 
    This function will run the mask over the image.
    If it finds instances such that there is enough overlap, in the final image it will return
    that overlapped section.
    '''

    mask_height = len(mask)
    mask_width = len(mask[0])
    img_height = len(img)
    img_width = len(img[0])

    result_img = np.array(img)
    result_img[:,:] = 255

    for i in range(0, img_height-mask_height, mask_height):
        for j in range(0, img_width-mask_width, mask_width):
            total_cells_here = mask_height * mask_width
            total_good_cells = 0
            for k in range(i, i+mask_height):
                for l in range(j, j+mask_width):
                    if (img[k][l] | mask[k-i][l-j]) == 0:
                        total_good_cells += 1
            if (total_good_cells / total_cells_here) >= 0.9:
                for k in range(i, i+mask_height):
                    for l in range(j, j+mask_width):
                        if (img[k][l] | mask[k-i][l-j]) == 0:
                            result_img[k][l] = 0
    
    return result_img


def main():
    cropped_img = cv.imread('cropped.png',0)
    ret, bin_image = cv.threshold(cropped_img, 155, 255, cv.THRESH_BINARY)
    mask = np.zeros((5,20), np.uint8)

    result = highlight_feature(mask, bin_image)
    cv.imwrite("result.png", result)

    STRAIGHT_LINES_COUNT = 0

    for i in range(len(result)):
        blackrun = 0
        for j in range(len(result[0])):
            if result[i][j] == 0:
                blackrun += 1
            else:
                if blackrun >= 20:
                    STRAIGHT_LINES_COUNT += 1
                blackrun = 0

    print('Straight line count =', STRAIGHT_LINES_COUNT)
    return STRAIGHT_LINES_COUNT

# if __name__ == '__main__':
#     main()
