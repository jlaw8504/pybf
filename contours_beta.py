#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to import a timelpase of Nup49-GFP signals
Created on Thu Aug 23 09:34:53 2018

@author: lawrimor
"""

import pybf
import cv2
import matplotlib.pyplot as plt

pybf.start_vm()
mat = pybf.im_compile('GFP1.stk')
trans = pybf.im_compile('Trans1.stk')
pybf.kill_vm()
image_array = pybf.deint_zt(mat, 1, 7, 21)
trans = pybf.deint_zt(trans, 1, 7, 21)
image_array = pybf.max_int_proj(image_array, 3)
trans = pybf.max_int_proj(trans, 3)
image_array8 = pybf.scale_8bit(image_array)
trans8 = pybf.scale_8bit(trans)
trans_plane = trans8[:,:,0]
plane = image_array8[:,:,0]
blur = cv2.medianBlur(plane, 9)
trans_blur = cv2.medianBlur(trans_plane, 9)
plt.imshow(blur, 'gray')
plt.title('Blurred Max Intensity Projection')
plt.waitforbuttonpress()
plt.close('all')
ret, otsu = cv2.threshold(blur, plane.min(), plane.max(),
                          cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(otsu)
plt.title('Binary Image')
plt.waitforbuttonpress()
plt.close('all')
_, contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)      
    cv2.rectangle(blur,(x,y),(x+w,y+h),(255,255,255),2)
    cv2.rectangle(trans_blur, (x,y),(x+w,y+h),(255,255,255),2)
    plt.imshow(blur, 'gray')
    plt.title('Height: ' + str(round(h*64.5)) + ', Width ' + str(round(w*64.5)))
#    crop = image_array[y:y+h, x:x+w, 0]
#    plt.imshow(crop, 'gray')
    plt.waitforbuttonpress()
    plt.close('all')
plt.imshow(trans_blur, 'gray')
plt.title('Labeled Trans Image')