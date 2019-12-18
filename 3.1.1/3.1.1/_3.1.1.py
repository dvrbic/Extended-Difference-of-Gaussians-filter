import cv2
import numpy as np
import math

img1 = cv2.imread('digital_orca_blurred.png',cv2.IMREAD_GRAYSCALE)

def gaussianblur(img,r,sig):

    #Gaussian blurring
    rangevar = r*2+1 #range of the kernel for the loop
    blurKernel = np.zeros((rangevar,rangevar)) #empty kernel of zeros, equivalent to the diameter of a 2d gaussian
     # blurring kernel creation
    for i in range(0, rangevar):
     for j in range(0, rangevar):
        z = float(((i - r) ** 2) + ((j - r) ** 2)) #simplifying part of the exponent into 1 variable for simplicity
        gauss_coef = float(1/(2*math.pi*(sig**2)))
        blurKernel[i, j] = gauss_coef*(math.exp((-1 * z) / (2 * (sig ** 2))))
    print(blurKernel) #makes the gaussian kernel visible

    #convolution function
    value = 0
    new_img = np.empty([img.shape[0], img.shape[1]], dtype=float) #creates an empty array as a float type to be compatible for convolution
    #traverses through the 2d range of the image
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if x < r or y < r or (x >= (img.shape[0]-r)) or (y >= (img.shape[1]-r)): #if the mask is out of range of the image
                new_img[x][y] = 0
                pass
            else:
                #goes through the range of the kernel mask
                for i in range(-1*r, r+1):
                    for j in range(-1*r, r+1):
                        #multiplies the coordinates of the mask with the image and sums them up to produce a new pixel intensity
                        value = value + (img[x+i, y+j]*blurKernel[r+i, r+j])
                new_img[x][y] = value
                value = 0
    return new_img

#XDoG
def XDoG(img, m):
    img_blur1 = gaussianblur(img, 3, 0.9) #blurred image with sigma
    img_blur2 = gaussianblur(img, 3, 1.2*0.9) #blurred image with sigma multiplied by k=(1.2)
    img_XDoG = np.empty([img_blur1.shape[0], img_blur1.shape[1]]) #empty image array with same dimensions as the blurred image
    for i in range(0, img_blur1.shape[0]):
        for j in range(0, img_blur1.shape[1]):
            img_XDoG[i][j] = img_blur1[i][j] + m*(img_blur1[i][j] - img_blur2[i][j]) #img_XDoG function
    return img_XDoG

final_img = XDoG(img1, 50)
#writes the new image
cv2.imwrite("XDoG_orca.png", final_img)
#displays new image
cv2.imshow('XDoG',final_img)
