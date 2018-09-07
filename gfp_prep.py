# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 16:06:22 2018

@author: joshl
"""

import pybf
import cv2


pybf.start_vm()
mat = pybf.im_compile('GFP1.stk')
image_array = pybf.deint_zt(mat, 1, 7, 21)
max_int = pybf.max_int_proj(image_array, 3)
plane = max_int[:,:,0]
plane8 = pybf.scale_8bit(plane)
blur = cv2.medianBlur(plane8, 9)
ret, otsu = cv2.threshold(blur, plane8.min(), plane8.max(),
                          cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
bounds = pybf.find_bounds(contours)
centers = pybf.find_centers(contours)
im_list = pybf.crop_square(image_array, centers)
for i, im in enumerate(im_list):
    filename = 'GFP1_' + str(i) + '.tif'
    pybf.write_hyper(im, filename)
