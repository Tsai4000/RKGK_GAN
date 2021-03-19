import cv2
import numpy as np
import os

for f in os.listdir('./Input'):

    img = cv2.imread('./Input/%s' % f, 1)
    # cv2.imshow('img', img)
    # img_shape = img.shape
    # print(img_shape)
    # h = img_shape[0]
    # w = img_shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    dst = 255 - gray
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # kernel = np.ones((3,3),np.uint8)
    # dilation = cv2.dilate(dst,kernel,iterations = 3)
    # res = cv2.resize(dilation, (28,28), interpolation=cv2.INTER_CUBIC)
    res = cv2.resize(dst, (80, 80), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('./InputReverseDilate/D%s' % f, res)
    res2 = cv2.rotate(res, cv2.ROTATE_180)
    cv2.imwrite('./InputReverseDilate/DR%s' % f, res2)
    res3 = cv2.flip(res, 1)
    cv2.imwrite('./InputReverseDilate/DF%s' % f, res3)
    res4 = cv2.flip(res2, 1)
    cv2.imwrite('./InputReverseDilate/DRF%s' % f, res4)


# dst = 255 - dst
# cv2.imshow('dst', dst)
# cv2.waitKey(0) #再度反白
